module Safetensors
  module Torch
    TYPES = {
      "F64" => :float64,
      "F32" => :float32,
      "F16" => :float16,
      "BF16" => :bfloat16,
      "I64" => :int64,
      "U64" => :uint64,
      "I32" => :int32,
      "U32" => :uint32,
      "I16" => :int16,
      "U16" => :uint16,
      "I8" => :int8,
      "U8" => :uint8,
      "BOOL" => :bool,
      "F8_E4M3" => :float8_e4m3fn,
      "F8_E5M2" => :float8_e5m2
    }

    class << self
      def save(tensors, metadata: nil)
        Safetensors.serialize(_flatten(tensors), metadata: metadata)
      end

      def save_file(tensors, filename, metadata: nil)
        Safetensors.serialize_file(_flatten(tensors), filename, metadata: metadata)
      end

      def load_file(filename, device: "cpu")
        result = {}
        Safetensors.safe_open(filename, framework: "torch", device: device) do |f|
          f.keys.each do |k|
            result[k] = f.get_tensor(k)
          end
        end
        result
      end

      def load(data)
        flat = Safetensors.deserialize(data)
        _view2torch(flat)
      end

      private

      def _find_shared_tensors(state_dict)
        # TODO
        []
      end

      def _getdtype(dtype_str)
        TYPES.fetch(dtype_str)
      end

      def _view2torch(safeview)
        result = {}
        safeview.each do |k, v|
          dtype = _getdtype(v["dtype"])
          options = ::Torch.send(:tensor_options, dtype: dtype)
          arr = ::Torch._from_blob_ref(v["data"], v["shape"], options)
          if Safetensors.big_endian?
            # TODO
            raise "not yet implemented"
          end
          result[k] = arr
        end
        result
      end

      def _tobytes(tensor, name)
        if tensor.layout != :strided
          raise ArgumentError, "You are trying to save a sparse tensor: `#{name}` which this library does not support. You can make it a dense tensor before saving with `.to_dense()` but be aware this might make a much larger file than needed."
        end

        if !tensor.contiguous?
          raise ArgumentError, "You are trying to save a non contiguous tensor: `#{name}` which is not allowed. It either means you are trying to save tensors which are reference of each other in which case it's recommended to save only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to pack it before saving."
        end

        if tensor.device != "cpu"
          # Moving tensor to cpu before saving
          tensor = tensor.to("cpu")
        end

        if Safetensors.big_endian?
          # TODO
          raise "not yet implemented"
        end

        tensor._data_str
      end

      def _flatten(tensors)
        if !tensors.is_a?(Hash)
          raise ArgumentError, "Expected a hash of [String, Torch::Tensor] but received #{tensors.class.name}"
        end

        invalid_tensors = []
        tensors.each do |k, v|
          if !v.is_a?(::Torch::Tensor)
            raise ArgumentError, "Key `#{k}` is invalid, expected Torch::Tensor but received #{v.class.name}"
          end

          if v.layout != :strided
            invalid_tensors << k
          end
        end
        if invalid_tensors.any?
          raise ArgumentError, "You are trying to save a sparse tensors: `#{invalid_tensors}` which this library does not support. You can make it a dense tensor before saving with `.to_dense()` but be aware this might make a much larger file than needed."
        end

        shared_pointers = _find_shared_tensors(tensors)
        failing = []
        shared_pointers.each do |names|
          if names.length > 1
            failing << names
          end
        end

        if failing.any?
          raise <<~MSG
            Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: #{failing}.
            A potential way to correctly save your model is to use `save_model`.
            More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
          MSG
        end

        tensors.to_h do |k, v|
          [
            k.is_a?(Symbol) ? k.to_s : k,
            {
              "dtype" => v.dtype.to_s,
              "shape" => v.shape,
              "data" => _tobytes(v, k)
            }
          ]
        end
      end
    end
  end
end
