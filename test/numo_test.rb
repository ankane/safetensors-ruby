require_relative "test_helper"

class NumoTest < Minitest::Test
  def test_save
    tensors = generate_tensors

    data = Safetensors::Numo.save(tensors)
    assert_tensors tensors, Safetensors::Numo.load(data)
  end

  def test_save_file
    tensors = generate_tensors

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Numo.save_file(tensors, path)
      assert_tensors tensors, Safetensors::Numo.load_file(path)
    end
  end

  def test_load_file_zero_rank
    tensors = {"weight1" => Numo::DFloat.cast(1)}

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Numo.save_file(tensors, path)
      assert_tensors tensors, Safetensors::Numo.load_file(path)
    end
  end

  def test_load_file_empty
    tensors = {"weight1" => Numo::DFloat.new([0]).rand}

    Dir.mktmpdir do |dir|
      path = File.join(dir, "model.safetensors")
      Safetensors::Numo.save_file(tensors, path)
      assert_tensors tensors, Safetensors::Numo.load_file(path)
    end
  end

  def test_save_symbol_keys
    tensors = {weight1: Numo::DFloat.new([3]).rand}

    data = Safetensors::Numo.save(tensors)
    assert_tensors tensors, Safetensors::Numo.load(data)
  end

  def test_save_invalid_object
    error = assert_raises(ArgumentError) do
      Safetensors::Numo.save(1)
    end
    assert_equal "Expected a hash of [String, Numo::NArray] but received Integer", error.message
  end

  def test_save_invalid_value
    error = assert_raises(ArgumentError) do
      Safetensors::Numo.save({"weight1" => 1})
    end
    assert_equal "Key `weight1` is invalid, expected Numo::NArray but received Integer", error.message
  end

  def test_load_invalid_value
    error = assert_raises(Safetensors::Error) do
      Safetensors::Numo.load("")
    end
    assert_includes error.message, "Error while deserializing"
  end

  private

  def generate_tensors
    {
      "weight1" => Numo::SFloat.new([1024, 1024]).rand,
      "weight2" => Numo::DFloat.new([1, 2, 3]).rand
    }
  end

  def assert_tensors(expected, actual)
    expected.each do |k, exp|
      act = actual[k.to_s]
      if exp.empty?
        assert_equal exp.size, act.size
      else
        assert exp.eq(act).all?
      end
      assert_equal exp.class, act.class
      assert_equal exp.shape, act.shape
    end
  end
end
