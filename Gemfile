source "https://rubygems.org"

gemspec

gem "rake"
gem "rake-compiler"
gem "minitest"
gem "torch-rb", platform: :ruby

# TODO remove when numo-narray > 0.9.2.1 is released
if Gem.win_platform?
  gem "numo-narray", github: "ruby-numo/numo-narray", ref: "421feddb46cac5145d69067fc1ac3ba3c434f668", require: "numo/narray"
else
  gem "numo-narray", require: "numo/narray"
end
