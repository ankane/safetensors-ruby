mod ruby;

use core::slice;
use magnus::{
    function, kwargs, method, prelude::*, r_hash::ForEach, Error, IntoValue, RArray, RHash,
    RModule, RString, Ruby, Symbol, TryConvert, Value,
};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{Dtype, Metadata, SafeTensors};
use safetensors::View;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ruby::GvlExt;

type RbResult<T> = Result<T, Error>;

#[derive(Clone, Debug)]
struct TensorSpec {
    dtype: Dtype,
    shape: Vec<usize>,
    data_ptr: u64,
    data_len: usize,
}

impl TensorSpec {
    fn new(dtype: &str, shape: Vec<usize>, data_ptr: u64, data_len: usize) -> RbResult<Self> {
        let dtype = parse_dtype_str(dtype)?;
        let mut shape = shape;
        // F4 packs two elements per byte; the safetensors header records the
        // logical element count, so double the last dim.
        if dtype == Dtype::F4 && !shape.is_empty() {
            let n = shape.len();
            shape[n - 1] = shape[n - 1].checked_mul(2).ok_or_else(|| {
                SafetensorError::new_err(format!(
                    "F4 last-dim {} doubled to logical shape overflows usize",
                    shape[n - 1]
                ))
            })?;
        }
        Ok(Self {
            dtype,
            shape,
            data_ptr,
            data_len,
        })
    }
}

impl View for &TensorSpec {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        let p = self.data_ptr as *const u8;
        // SAFETY: validated by the caller; see the struct-level safety note.
        unsafe {
            let slice = slice::from_raw_parts(p, self.data_len);
            Cow::Borrowed(slice)
        }
    }

    fn data_len(&self) -> usize {
        self.data_len
    }
}

fn parse_dtype_str(dtype: &str) -> RbResult<Dtype> {
    Ok(match dtype {
        "bool" => Dtype::BOOL,
        "int8" => Dtype::I8,
        "uint8" => Dtype::U8,
        "int16" => Dtype::I16,
        "uint16" => Dtype::U16,
        "int32" => Dtype::I32,
        "uint32" => Dtype::U32,
        "int64" => Dtype::I64,
        "uint64" => Dtype::U64,
        "float16" => Dtype::F16,
        "float32" => Dtype::F32,
        "float64" => Dtype::F64,
        "bfloat16" => Dtype::BF16,
        "float8_e4m3fn" => Dtype::F8_E4M3,
        "float8_e4m3fnuz" => Dtype::F8_E4M3FNUZ,
        "float8_e5m2" => Dtype::F8_E5M2,
        "float8_e5m2fnuz" => Dtype::F8_E5M2FNUZ,
        "float8_e8m0fnu" => Dtype::F8_E8M0,
        "float4_e2m1fn_x2" => Dtype::F4,
        "complex64" => Dtype::C64,
        other => {
            return Err(SafetensorError::new_err(format!(
                "Unknown dtype {other:?}. Supported dtypes: bool, int8, uint8, int16, uint16, \
                 int32, uint32, int64, uint64, float16, float32, float64, bfloat16, \
                 float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz, float8_e8m0fnu, \
                 float4_e2m1fn_x2, complex64",
            )));
        }
    })
}

fn prepare(tensor_dict: &RHash) -> RbResult<HashMap<String, TensorSpec>> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    tensor_dict.foreach(|tensor_name: String, tensor_desc: RHash| {
        let shape = Vec::<usize>::try_convert(tensor_desc.get("shape").ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `shape` in {tensor_desc:?}"))
        })?)?;
        let rbdata = tensor_desc.get("data").ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `data` in {tensor_desc:?}"))
        })?;

        let rbdtype = tensor_desc.get("dtype").ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
        })?;

        let dtype = String::try_convert(rbdtype)?;

        let rs = RString::try_convert(rbdata)?;
        // SAFETY: No context switching between threads in native extensions
        // so the string will not be modified (or garbage collected)
        // while the reference is held. Also, the string is a private copy.
        let slice = unsafe { rs.as_slice() };

        let tensor = TensorSpec::new(dtype.as_ref(), shape, slice.as_ptr() as u64, slice.len())
            .map_err(|e| SafetensorError::new_err(format!("Error preparing tensor view: {e:?}")))?;
        tensors.insert(tensor_name, tensor);

        Ok(ForEach::Continue)
    })?;
    Ok(tensors)
}

fn serialize(
    ruby: &Ruby,
    tensor_dict: RHash,
    metadata: Option<HashMap<String, String>>,
) -> RbResult<RString> {
    let tensors = prepare(&tensor_dict)?;
    let metadata_map = metadata.map(HashMap::from_iter);
    let out = ruby
        .detach(|| safetensors::tensor::serialize(&tensors, metadata_map))
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?;
    let rbbytes = ruby.str_from_slice(&out);
    Ok(rbbytes)
}

fn serialize_file(
    ruby: &Ruby,
    tensor_dict: RHash,
    filename: PathBuf,
    metadata: Option<HashMap<String, String>>,
) -> RbResult<()> {
    let tensors = prepare(&tensor_dict)?;
    ruby.detach(|| safetensors::tensor::serialize_to_file(&tensors, metadata, filename.as_path()))
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?;
    Ok(())
}

fn deserialize(ruby: &Ruby, bytes: RString) -> RbResult<RArray> {
    let safetensor = SafeTensors::deserialize(unsafe { bytes.as_slice() })
        .map_err(|e| SafetensorError::new_err(format!("Error while deserializing: {e:?}")))?;

    let tensors = safetensor.tensors();
    let items = ruby.ary_new_capa(tensors.len());

    for (tensor_name, tensor) in tensors {
        let rbshape = ruby.ary_from_vec(tensor.shape().to_vec());
        let rbdtype = format!("{:?}", tensor.dtype());

        let rbdata = ruby.str_from_slice(tensor.data());

        let map = ruby.hash_new();
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

impl fmt::Display for Framework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Framework::Pytorch => "torch",
            Framework::Numo => "numo",
        })
    }
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
    Xla(usize),
    Mlu(usize),
    Hpu(usize),
    Anonymous(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Device::Cpu => write!(f, "cpu"),
            Device::Mps => write!(f, "mps"),
            Device::Cuda(index) => write!(f, "cuda:{index}"),
            Device::Npu(index) => write!(f, "npu:{index}"),
            Device::Xpu(index) => write!(f, "xpu:{index}"),
            Device::Xla(index) => write!(f, "xla:{index}"),
            Device::Mlu(index) => write!(f, "mlu:{index}"),
            Device::Hpu(index) => write!(f, "hpu:{index}"),
            Device::Anonymous(index) => write!(f, "{index}"),
        }
    }
}

/// Parsing the device index.
fn parse_device(name: &str) -> RbResult<usize> {
    let tokens: Vec<_> = name.split(':').collect();
    if tokens.len() == 2 {
        Ok(tokens[1].parse().map_err(SafetensorError::parse)?)
    } else {
        Err(SafetensorError::new_err(format!(
            "device {name} is invalid"
        )))
    }
}

impl TryConvert for Device {
    fn try_convert(ob: Value) -> RbResult<Self> {
        if let Ok(name) = String::try_convert(ob) {
            match name.as_str() {
                "cpu" => Ok(Device::Cpu),
                "cuda" => Ok(Device::Cuda(0)),
                "mps" => Ok(Device::Mps),
                "npu" => Ok(Device::Npu(0)),
                "xpu" => Ok(Device::Xpu(0)),
                "xla" => Ok(Device::Xla(0)),
                "mlu" => Ok(Device::Mlu(0)),
                "hpu" => Ok(Device::Hpu(0)),
                name if name.starts_with("cuda:") => parse_device(name).map(Device::Cuda),
                name if name.starts_with("npu:") => parse_device(name).map(Device::Npu),
                name if name.starts_with("xpu:") => parse_device(name).map(Device::Xpu),
                name if name.starts_with("xla:") => parse_device(name).map(Device::Xla),
                name if name.starts_with("mlu:") => parse_device(name).map(Device::Mlu),
                name if name.starts_with("hpu:") => parse_device(name).map(Device::Hpu),
                name => Err(SafetensorError::new_err(format!(
                    "device {name} is invalid"
                ))),
            }
        } else if let Ok(number) = usize::try_convert(ob) {
            Ok(Device::Anonymous(number))
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
            Device::Xla(n) => format!("xla:{n}").into_value_with(ruby),
            Device::Mlu(n) => format!("mlu:{n}").into_value_with(ruby),
            Device::Hpu(n) => format!("hpu:{n}").into_value_with(ruby),
            Device::Anonymous(n) => n.into_value_with(ruby),
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
            SafetensorError::new_err(format!("No such file or directory: {}", filename.display()))
        })?;
        let device = device.unwrap_or(Device::Cpu);

        if device != Device::Cpu && framework != Framework::Pytorch {
            return Err(SafetensorError::new_err(format!(
                "Device {device} is not support for framework {framework}",
            )));
        }

        // SAFETY: Mmap is used to prevent allocating in Rust
        // before making a copy within Ruby.
        let buffer = unsafe { MmapOptions::new().map(&file).map_err(SafetensorError::io)? };

        let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|e| {
            SafetensorError::new_err(format!("Error while deserializing header: {e}"))
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

    pub fn get_tensor(&self, ruby: &Ruby, name: &str) -> RbResult<Value> {
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}"))
        })?;

        match &self.storage.as_ref() {
            Storage::Mmap(mmap) => {
                let data =
                    &mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

                let array: Value = ruby.str_from_slice(data).as_value();

                create_tensor(
                    ruby,
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

    pub fn get_tensor(ruby: &Ruby, rb_self: &Self, name: String) -> RbResult<Value> {
        rb_self.inner()?.get_tensor(ruby, &name)
    }
}

fn create_tensor(
    ruby: &Ruby,
    framework: &Framework,
    dtype: Dtype,
    shape: &[usize],
    array: Value,
    device: &Device,
) -> RbResult<Value> {
    let (module, is_numo): (RModule, bool) = match framework {
        Framework::Pytorch => (
            ruby.class_object()
                .const_get("Torch")
                .map_err(|_| SafetensorError::new_err("Torch not loaded".into()))?,
            false,
        ),
        Framework::Numo => (
            ruby.class_object()
                .const_get("Numo")
                .map_err(|_| SafetensorError::new_err("Numo not loaded".into()))?,
            true,
        ),
    };

    let dtype = get_rbdtype(ruby, module, dtype, is_numo)?;
    let shape = shape.to_vec();
    let tensor: Value = match framework {
        Framework::Pytorch => {
            let options: Value = module.funcall(
                "tensor_options",
                (kwargs!("dtype" => dtype, "device" => device.clone()),),
            )?;
            module.funcall("_from_blob_ref", (array, shape, options))?
        }
        Framework::Numo => {
            let class: Value = module.funcall("const_get", (dtype,))?;
            class.funcall("from_binary", (array, shape))?
        }
    };
    Ok(tensor)
}

fn get_rbdtype(ruby: &Ruby, _module: RModule, dtype: Dtype, is_numo: bool) -> RbResult<Symbol> {
    let dtype: Symbol = if is_numo {
        match dtype {
            Dtype::F64 => ruby.to_symbol("DFloat"),
            Dtype::F32 => ruby.to_symbol("SFloat"),
            Dtype::U64 => ruby.to_symbol("UInt64"),
            Dtype::I64 => ruby.to_symbol("Int64"),
            Dtype::U32 => ruby.to_symbol("UInt32"),
            Dtype::I32 => ruby.to_symbol("Int32"),
            Dtype::U16 => ruby.to_symbol("UInt16"),
            Dtype::I16 => ruby.to_symbol("Int16"),
            Dtype::U8 => ruby.to_symbol("UInt8"),
            Dtype::I8 => ruby.to_symbol("Int8"),
            dtype => {
                return Err(SafetensorError::new_err(format!(
                    "Dtype not understood: {dtype:?}"
                )))
            }
        }
    } else {
        match dtype {
            Dtype::F64 => ruby.to_symbol("float64"),
            Dtype::F32 => ruby.to_symbol("float32"),
            Dtype::BF16 => ruby.to_symbol("bfloat16"),
            Dtype::F16 => ruby.to_symbol("float16"),
            Dtype::U64 => ruby.to_symbol("uint64"),
            Dtype::I64 => ruby.to_symbol("int64"),
            Dtype::U32 => ruby.to_symbol("uint32"),
            Dtype::I32 => ruby.to_symbol("int32"),
            Dtype::U16 => ruby.to_symbol("uint16"),
            Dtype::I16 => ruby.to_symbol("int16"),
            Dtype::U8 => ruby.to_symbol("uint8"),
            Dtype::I8 => ruby.to_symbol("int8"),
            Dtype::BOOL => ruby.to_symbol("bool"),
            Dtype::F8_E4M3 => ruby.to_symbol("float8_e4m3fn"),
            Dtype::F8_E5M2 => ruby.to_symbol("float8_e5m2"),
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

#[magnus::init(name = "safetensors")]
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
