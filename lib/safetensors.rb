# ext
begin
  require "safetensors/#{RUBY_VERSION.to_f}/safetensors"
rescue LoadError
  require "safetensors/safetensors"
end

# modules
require_relative "safetensors/numo"
require_relative "safetensors/torch"
require_relative "safetensors/version"

module Safetensors
  class Error < StandardError; end

  def self.serialize(tensor_dict, metadata: nil)
    _serialize(tensor_dict, metadata)
  end

  def self.serialize_file(tensor_dict, filename, metadata: nil)
    _serialize_file(tensor_dict, filename, metadata)
  end

  def self.safe_open(filename, framework:, device: "cpu")
    f = SafeOpen.new(filename, framework, device)
    if block_given?
      yield f
    else
      f
    end
  end

  # private
  def self.big_endian?
    [1].pack("i") == [1].pack("i!>")
  end
end
