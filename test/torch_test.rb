require_relative "test_helper"

class TorchTest < Minitest::Test
  def test_save
    tensors = generate_tensors

    data = Safetensors::Torch.save(tensors)
    assert_tensors tensors, Safetensors::Torch.load(data)
  end

  def test_save_file
    tensors = generate_tensors

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Torch.save_file(tensors, path)
      assert_tensors tensors, Safetensors::Torch.load_file(path)
    end
  end

  def test_safe_open
    tensors = generate_tensors
    metadata = {"hello" => "world"}

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Torch.save_file(tensors, path, metadata: metadata)

      new_tensors = {}
      Safetensors.safe_open(path, framework: "torch", device: "cpu") do |f|
        assert_equal metadata, f.metadata
        f.keys.each do |key|
          new_tensors[key] = f.get_tensor(key)
        end
      end
      assert_tensors tensors, new_tensors
    end
  end

  def test_zero_rank
    tensors = {"weight1" => Torch.tensor(1)}

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Torch.save_file(tensors, path)
      assert_tensors tensors, Safetensors::Torch.load_file(path)
    end
  end

  def test_empty
    tensors = {"weight1" => Torch.rand([0])}

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Torch.save_file(tensors, path)
      assert_tensors tensors, Safetensors::Torch.load_file(path)
    end
  end

  def test_symbol_keys
    tensors = {weight1: Torch.rand([3])}

    data = Safetensors::Torch.save(tensors)
    assert_tensors tensors, Safetensors::Torch.load(data)
  end

  def test_invalid_object
    error = assert_raises(ArgumentError) do
      Safetensors::Torch.save(1)
    end
    assert_equal "Expected a hash of [String, Torch::Tensor] but received Integer", error.message
  end

  def test_invalid_value
    error = assert_raises(ArgumentError) do
      Safetensors::Torch.save({"weight1" => 1})
    end
    assert_equal "Key `weight1` is invalid, expected Torch::Tensor but received Integer", error.message
  end

  private

  def generate_tensors
    {
      "weight1" => Torch.rand([1024, 1024]),
      "weight2" => Torch.rand([1, 2, 3], dtype: :float64)
    }
  end

  def assert_tensors(expected, actual)
    expected.each do |k, exp|
      act = actual[k.to_s]
      assert Torch.equal(exp, act)
      assert_equal exp.device, act.device
      assert_equal exp.dtype, act.dtype
      assert_equal exp.shape, act.shape
    end
  end
end
