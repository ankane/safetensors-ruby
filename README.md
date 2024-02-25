# Safetensors Ruby

:slightly_smiling_face: Simple, [safe way](https://github.com/huggingface/safetensors) to store and distribute tensors

Supports [Torch.rb](https://github.com/ankane/torch.rb) and [Numo](https://github.com/ruby-numo/numo-narray)

[![Build Status](https://github.com/ankane/safetensors-ruby/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/safetensors-ruby/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem "safetensors"
```

## Getting Started

Save tensors

```ruby
tensors = {
  "weight1" => Torch.zeros([1024, 1024]),
  "weight2" => Torch.zeros([1024, 1024])
}
Safetensors::Torch.save_file(tensors, "model.safetensors")
```

Load tensors

```ruby
tensors = Safetensors::Torch.load_file("model.safetensors")
# or
tensors = {}
Safetensors.safe_open("model.safetensors", framework: "torch", device: "cpu") do |f|
  f.keys.each do |key|
    tensors[key] = f.get_tensor(key)
  end
end
```

## API

This library follows the [Safetensors Python API](https://huggingface.co/docs/safetensors/index). You can follow Python tutorials and convert the code to Ruby in many cases. Feel free to open an issue if you run into problems.

## History

View the [changelog](https://github.com/ankane/safetensors-ruby/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/safetensors-ruby/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/safetensors-ruby/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/safetensors-ruby.git
cd safetensors-ruby
bundle install
bundle exec rake compile
bundle exec rake test
```
