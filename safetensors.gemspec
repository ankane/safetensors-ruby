require_relative "lib/safetensors/version"

Gem::Specification.new do |spec|
  spec.name          = "safetensors"
  spec.version       = Safetensors::VERSION
  spec.summary       = "Simple, safe way to store and distribute tensors"
  spec.homepage      = "https://github.com/ankane/safetensors-ruby"
  spec.license       = "Apache-2.0"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{ext,lib}/**/*", "Cargo.*"]
  spec.require_path  = "lib"
  spec.extensions    = ["ext/safetensors/extconf.rb"]

  spec.required_ruby_version = ">= 3.2"

  spec.add_dependency "rb_sys"
end
