use magnus::{
    function, kwargs, method, prelude::*, r_hash::ForEach, Error, IntoValue, RArray, RHash,
    RModule, RString, Ruby, Symbol, TryConvert, Value,
};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

type RbResult<T> = Result<T, Error>;

fn prepare(tensor_dict: &RHash) -> RbResult<HashMap<String, TensorView<'_>>> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    tensor_dict.foreach(|tensor_name: String, tensor_desc: RHash| {
        let mut shape: Option<Vec<usize>> = None;
        let mut dtype: Option<Dtype> = None;
        let mut data: Option<(*const u8, usize)> = None;

        tensor_desc.foreach(|key: String, value: Value| {
            match key.as_str() {
                "shape" => shape = Some(Vec::try_convert(value)?),
                "dtype" => {
                    let value = String::try_convert(value)?;
                    dtype = match value.as_str() {
                        "bool" => Some(Dtype::BOOL),
                        "int8" => Some(Dtype::I8),
                        "uint8" => Some(Dtype::U8),
                        "int16" => Some(Dtype::I16),
                        "uint16" => Some(Dtype::U16),
                        "int32" => Some(Dtype::I32),
                        "uint32" => Some(Dtype::U32),
                        "int64" => Some(Dtype::I64),
                        "uint64" => Some(Dtype::U64),
                        "float16" => Some(Dtype::F16),
                        "float32" => Some(Dtype::F32),
                        "float64" => Some(Dtype::F64),
                        "bfloat16" => Some(Dtype::BF16),
                        "float8_e4m3fn" => Some(Dtype::F8_E4M3),
                        "float8_e5m2" => Some(Dtype::F8_E5M2),
                        dtype_str => {
                            return Err(SafetensorError::new_err(format!(
                                "dtype {dtype_str} is not covered",
                            )));
                        }
                    }
                }
                "data" => {
                    let rs = RString::try_convert(value)?;
                    // SAFETY: No context switching between threads in native extensions
                    // so the string will not be modified (or garbage collected)
                    // while the reference is held. Also, the string is a private copy.
                    let slice = unsafe { rs.as_slice() };
                    data = Some((slice.as_ptr(), slice.len()));
                }
                _ => println!("Ignored unknown kwarg option {key}"),
            };

            Ok(ForEach::Continue)
        })?;
        let shape = shape.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `shape` in {tensor_desc:?}"))
        })?;
        let dtype = dtype.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
        })?;
        let data = data.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `data` in {tensor_desc:?}"))
        })?;
        // SAFETY: See comment above.
        let data = unsafe { std::slice::from_raw_parts(data.0, data.1) };
        let tensor = TensorView::new(dtype, shape, data)
            .map_err(|e| SafetensorError::new_err(format!("Error preparing tensor view: {e:?}")))?;
        tensors.insert(tensor_name, tensor);

        Ok(ForEach::Continue)
    })?;
    Ok(tensors)
}

fn serialize(tensor_dict: RHash, metadata: Option<HashMap<String, String>>) -> RbResult<RString> {
    let tensors = prepare(&tensor_dict)?;
    let metadata_map = metadata.map(HashMap::from_iter);
    let out = safetensors::tensor::serialize(&tensors, metadata_map)
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?;
    let rbbytes = RString::from_slice(&out);
    Ok(rbbytes)
}

fn serialize_file(
    tensor_dict: RHash,
    filename: PathBuf,
    metadata: Option<HashMap<String, String>>,
) -> RbResult<()> {
    let tensors = prepare(&tensor_dict)?;
    safetensors::tensor::serialize_to_file(&tensors, metadata, filename.as_path())
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?;
    Ok(())
}

fn deserialize(bytes: RString) -> RbResult<RArray> {
    let safetensor = SafeTensors::deserialize(unsafe { bytes.as_slice() })
        .map_err(|e| SafetensorError::new_err(format!("Error while deserializing: {e:?}")))?;

    let tensors = safetensor.tensors();
    let items = RArray::with_capacity(tensors.len());

    for (tensor_name, tensor) in tensors {
        let rbshape = RArray::from_vec(tensor.shape().to_vec());
        let rbdtype = format!("{:?}", tensor.dtype());

        let rbdata = RString::from_slice(tensor.data());

        let map = RHash::new();
        map.aset("shape", rbshape)?;
        map.aset("dtype", rbdtype)?;
        map.aset("data", rbdata)?;

        items.push((tensor_name, map))?;
    }
    Ok(items)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Framework {
    Pytorch,
    Numo,
}

impl TryConvert for Framework {
    fn try_convert(ob: Value) -> RbResult<Self> {
        let name: String = String::try_convert(ob)?;
        match &name[..] {
            "pt" => Ok(Framework::Pytorch),
            "torch" => Ok(Framework::Pytorch),
            "pytorch" => Ok(Framework::Pytorch),

            "nm" => Ok(Framework::Numo),
            "numo" => Ok(Framework::Numo),

            name => Err(SafetensorError::new_err(format!(
                "framework {name} is invalid"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Device {
    Cpu,
    Cuda(usize),
    Mps,
    Npu(usize),
    Xpu(usize),
}

impl TryConvert for Device {
    fn try_convert(ob: Value) -> RbResult<Self> {
        if let Ok(name) = String::try_convert(ob) {
            match &name[..] {
                "cpu" => Ok(Device::Cpu),
                "cuda" => Ok(Device::Cuda(0)),
                "mps" => Ok(Device::Mps),
                "npu" => Ok(Device::Npu(0)),
                "xpu" => Ok(Device::Xpu(0)),
                name if name.starts_with("cuda:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse().map_err(SafetensorError::parse)?;
                        Ok(Device::Cuda(device))
                    } else {
                        Err(SafetensorError::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
                name if name.starts_with("npu:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse().map_err(SafetensorError::parse)?;
                        Ok(Device::Npu(device))
                    } else {
                        Err(SafetensorError::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
                name if name.starts_with("xpu:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse().map_err(SafetensorError::parse)?;
                        Ok(Device::Xpu(device))
                    } else {
                        Err(SafetensorError::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
                name => Err(SafetensorError::new_err(format!(
                    "device {name} is invalid"
                ))),
            }
        } else if let Ok(number) = usize::try_convert(ob) {
            Ok(Device::Cuda(number))
        } else {
            Err(SafetensorError::new_err(format!("device {ob} is invalid")))
        }
    }
}

impl IntoValue for Device {
    fn into_value_with(self, ruby: &Ruby) -> Value {
        match self {
            Device::Cpu => "cpu".into_value_with(ruby),
            Device::Cuda(n) => format!("cuda:{n}").into_value_with(ruby),
            Device::Mps => "mps".into_value_with(ruby),
            Device::Npu(n) => format!("npu:{n}").into_value_with(ruby),
            Device::Xpu(n) => format!("xpu:{n}").into_value_with(ruby),
        }
    }
}

enum Storage {
    Mmap(Mmap),
}

struct Open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    storage: Arc<Storage>,
}

impl Open {
    fn new(filename: PathBuf, framework: Framework, device: Option<Device>) -> RbResult<Self> {
        let file = File::open(&filename).map_err(|_| {
            SafetensorError::new_err(format!("No such file or directory: {filename:?}"))
        })?;
        let device = device.unwrap_or(Device::Cpu);

        if device != Device::Cpu && framework != Framework::Pytorch {
            return Err(SafetensorError::new_err(format!(
                "Device {device:?} is not support for framework {framework:?}",
            )));
        }

        // SAFETY: Mmap is used to prevent allocating in Rust
        // before making a copy within Ruby.
        let buffer = unsafe { MmapOptions::new().map(&file).map_err(SafetensorError::io)? };

        let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|e| {
            SafetensorError::new_err(format!("Error while deserializing header: {e:?}"))
        })?;

        let offset = n + 8;

        let storage = Storage::Mmap(buffer);

        let storage = Arc::new(storage);

        Ok(Self {
            metadata,
            offset,
            framework,
            device,
            storage,
        })
    }

    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        self.metadata.metadata().clone()
    }

    pub fn keys(&self) -> RbResult<Vec<String>> {
        let mut keys: Vec<String> = self.metadata.tensors().keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    pub fn get_tensor(&self, name: &str) -> RbResult<Value> {
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}",))
        })?;

        match &self.storage.as_ref() {
            Storage::Mmap(mmap) => {
                let data =
                    &mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

                let array: Value = RString::from_slice(data).into_value();

                create_tensor(
                    &self.framework,
                    info.dtype,
                    &info.shape,
                    array,
                    &self.device,
                )
            }
        }
    }
}

#[magnus::wrap(class = "Safetensors::SafeOpen")]
struct SafeOpen {
    inner: Option<Open>,
}

impl SafeOpen {
    fn inner(&self) -> RbResult<&Open> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))?;
        Ok(inner)
    }
}

impl SafeOpen {
    pub fn new(filename: PathBuf, framework: Framework, device: Option<Device>) -> RbResult<Self> {
        let inner = Some(Open::new(filename, framework, device)?);
        Ok(Self { inner })
    }

    pub fn metadata(&self) -> RbResult<Option<HashMap<String, String>>> {
        Ok(self.inner()?.metadata())
    }

    pub fn keys(&self) -> RbResult<Vec<String>> {
        self.inner()?.keys()
    }

    pub fn get_tensor(&self, name: String) -> RbResult<Value> {
        self.inner()?.get_tensor(&name)
    }
}

fn create_tensor(
    framework: &Framework,
    dtype: Dtype,
    shape: &[usize],
    array: Value,
    device: &Device,
) -> RbResult<Value> {
    let ruby = Ruby::get().unwrap();
    let (module, is_numo): (RModule, bool) = match framework {
        Framework::Pytorch => (
            ruby.class_object()
                .const_get("Torch")
                .map_err(|_| SafetensorError::new_err("Torch not loaded".into()))?,
            false,
        ),
        _ => (
            ruby.class_object()
                .const_get("Numo")
                .map_err(|_| SafetensorError::new_err("Numo not loaded".into()))?,
            true,
        ),
    };

    let dtype = get_rbdtype(module, dtype, is_numo)?;
    let shape = shape.to_vec();
    let tensor: Value = match framework {
        Framework::Pytorch => {
            let options: Value = module.funcall(
                "tensor_options",
                (kwargs!("dtype" => dtype, "device" => device.clone()),),
            )?;
            module.funcall("_from_blob_ref", (array, shape, options))?
        }
        _ => {
            let class: Value = module.funcall("const_get", (dtype,))?;
            class.funcall("from_binary", (array, shape))?
        }
    };
    Ok(tensor)
}

fn get_rbdtype(_module: RModule, dtype: Dtype, is_numo: bool) -> RbResult<Value> {
    let dtype: Value = if is_numo {
        match dtype {
            Dtype::F64 => Symbol::new("DFloat").into_value(),
            Dtype::F32 => Symbol::new("SFloat").into_value(),
            Dtype::U64 => Symbol::new("UInt64").into_value(),
            Dtype::I64 => Symbol::new("Int64").into_value(),
            Dtype::U32 => Symbol::new("UInt32").into_value(),
            Dtype::I32 => Symbol::new("Int32").into_value(),
            Dtype::U16 => Symbol::new("UInt16").into_value(),
            Dtype::I16 => Symbol::new("Int16").into_value(),
            Dtype::U8 => Symbol::new("UInt8").into_value(),
            Dtype::I8 => Symbol::new("Int8").into_value(),
            dtype => {
                return Err(SafetensorError::new_err(format!(
                    "Dtype not understood: {dtype:?}"
                )))
            }
        }
    } else {
        match dtype {
            Dtype::F64 => Symbol::new("float64").into_value(),
            Dtype::F32 => Symbol::new("float32").into_value(),
            Dtype::BF16 => Symbol::new("bfloat16").into_value(),
            Dtype::F16 => Symbol::new("float16").into_value(),
            Dtype::U64 => Symbol::new("uint64").into_value(),
            Dtype::I64 => Symbol::new("int64").into_value(),
            Dtype::U32 => Symbol::new("uint32").into_value(),
            Dtype::I32 => Symbol::new("int32").into_value(),
            Dtype::U16 => Symbol::new("uint16").into_value(),
            Dtype::I16 => Symbol::new("int16").into_value(),
            Dtype::U8 => Symbol::new("uint8").into_value(),
            Dtype::I8 => Symbol::new("int8").into_value(),
            Dtype::BOOL => Symbol::new("bool").into_value(),
            Dtype::F8_E4M3 => Symbol::new("float8_e4m3fn").into_value(),
            Dtype::F8_E5M2 => Symbol::new("float8_e5m2").into_value(),
            dtype => {
                return Err(SafetensorError::new_err(format!(
                    "Dtype not understood: {dtype:?}"
                )))
            }
        }
    };
    Ok(dtype)
}

struct SafetensorError {}

impl SafetensorError {
    fn new_err(message: String) -> Error {
        let class = Ruby::get()
            .unwrap()
            .class_object()
            .const_get::<_, RModule>("Safetensors")
            .unwrap()
            .const_get("Error")
            .unwrap();
        Error::new(class, message)
    }

    fn io(err: std::io::Error) -> Error {
        Self::new_err(err.to_string())
    }

    fn parse(err: std::num::ParseIntError) -> Error {
        Self::new_err(err.to_string())
    }
}

#[magnus::init]
fn init(ruby: &Ruby) -> RbResult<()> {
    let module = ruby.define_module("Safetensors")?;
    module.define_singleton_method("_serialize", function!(serialize, 2))?;
    module.define_singleton_method("_serialize_file", function!(serialize_file, 3))?;
    module.define_singleton_method("deserialize", function!(deserialize, 1))?;

    let class = module.define_class("SafeOpen", ruby.class_object())?;
    class.define_singleton_method("new", function!(SafeOpen::new, 3))?;
    class.define_method("metadata", method!(SafeOpen::metadata, 0))?;
    class.define_method("keys", method!(SafeOpen::keys, 0))?;
    class.define_method("get_tensor", method!(SafeOpen::get_tensor, 1))?;

    Ok(())
}
