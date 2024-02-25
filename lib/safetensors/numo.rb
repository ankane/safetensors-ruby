module Safetensors
  module Numo
    DTYPES = {
      "DFloat" => "float64",
      "SFloat" => "float32"
    }

    TYPES = {
      "F64" => :DFloat,
      "F32" => :SFloat,
      "I64" => :Int64,
      "U64" => :UInt64,
      "I32" => :Int32,
      "U32" => :UInt32,
      "I16" => :Int16,
      "U16" => :UInt16,
      "I8" => :Int8,
      "U8" => :UInt8
    }

    class << self
      def save(tensor_dict, metadata: nil)
        Safetensors.serialize(_flatten(tensor_dict), metadata: metadata)
      end

      def save_file(tensor_dict, filename, metadata: nil)
        Safetensors.serialize_file(_flatten(tensor_dict), filename, metadata: metadata)
      end

      def load(data)
        flat = Safetensors.deserialize(data)
        _view2numo(flat)
      end

      def load_file(filename)
        result = {}
        Safetensors.safe_open(filename, framework: "numo") do |f|
          f.keys.each do |k|
            result[k] = f.get_tensor(k)
          end
        end
        result
      end

      private

      def _flatten(tensors)
        if !tensors.is_a?(Hash)
          raise ArgumentError, "Expected a hash of [String, Numo::NArray] but received #{tensors.class.name}"
        end

        tensors.each do |k, v|
          if !v.is_a?(::Numo::NArray)
            raise ArgumentError, "Key `#{k}` is invalid, expected Numo::NArray but received #{v.class.name}"
          end
        end

        tensors.to_h do |k, v|
          [
            k.is_a?(Symbol) ? k.to_s : k,
            {
              "dtype" => DTYPES.fetch(v.class.name.split("::").last),
              "shape" => v.shape,
              "data" => _tobytes(v)
            }
          ]
        end
      end

      def _tobytes(tensor)
        if Safetensors.big_endian?
          raise "Not yet implemented"
        end

        tensor.to_binary
      end

      def _getdtype(dtype_str)
        TYPES.fetch(dtype_str)
      end

      def _view2numo(safeview)
        result = {}
        safeview.each do |k, v|
          dtype = _getdtype(v["dtype"])
          arr = ::Numo.const_get(dtype).from_binary(v["data"], v["shape"])
          result[k] = arr
        end
        result
      end
    end
  end
end
