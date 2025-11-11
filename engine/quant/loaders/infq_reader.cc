#include "infeng/quant/infq_reader.h"

#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace infeng::quant {

namespace {

struct JsonValue {
  enum class Type { Null, Object, Array, String, Number, Bool };
  Type type{Type::Null};
  std::map<std::string, JsonValue> object;
  std::vector<JsonValue> array;
  std::string string_value;
  double number_value{0.0};
  bool bool_value{false};
};

void skip_ws(const std::string& s, std::size_t& pos) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
}

std::string parse_string(const std::string& s, std::size_t& pos) {
  if (s[pos] != '"') {
    throw std::runtime_error("JSON parse error: expected string");
  }
  ++pos;
  std::string result;
  while (pos < s.size()) {
    char ch = s[pos++];
    if (ch == '"') {
      return result;
    }
    if (ch == '\\') {
      if (pos >= s.size()) {
        throw std::runtime_error("JSON parse error: invalid escape");
      }
      char esc = s[pos++];
      switch (esc) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default:
          throw std::runtime_error("JSON parse error: unsupported escape");
      }
    } else {
      result.push_back(ch);
    }
  }
  throw std::runtime_error("JSON parse error: unterminated string");
}

JsonValue parse_value(const std::string& s, std::size_t& pos);

JsonValue parse_object(const std::string& s, std::size_t& pos) {
  JsonValue value;
  value.type = JsonValue::Type::Object;
  ++pos;  // skip '{'
  skip_ws(s, pos);
  if (pos < s.size() && s[pos] == '}') {
    ++pos;
    return value;
  }
  while (pos < s.size()) {
    skip_ws(s, pos);
    std::string key = parse_string(s, pos);
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != ':') {
      throw std::runtime_error("JSON parse error: expected ':'");
    }
    ++pos;
    skip_ws(s, pos);
    JsonValue child = parse_value(s, pos);
    value.object.emplace(std::move(key), std::move(child));
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      break;
    }
    throw std::runtime_error("JSON parse error: expected ',' or '}'");
  }
  return value;
}

JsonValue parse_array(const std::string& s, std::size_t& pos) {
  JsonValue value;
  value.type = JsonValue::Type::Array;
  ++pos;  // skip '['
  skip_ws(s, pos);
  if (pos < s.size() && s[pos] == ']') {
    ++pos;
    return value;
  }
  while (pos < s.size()) {
    JsonValue element = parse_value(s, pos);
    value.array.push_back(std::move(element));
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < s.size() && s[pos] == ']') {
      ++pos;
      break;
    }
    throw std::runtime_error("JSON parse error: expected ',' or ']'");
  }
  return value;
}

JsonValue parse_number(const std::string& s, std::size_t& pos) {
  const std::size_t start = pos;
  if (s[pos] == '-' ) {
    ++pos;
  }
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos < s.size() && s[pos] == '.') {
    ++pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
      ++pos;
    }
  }
  const double value = std::stod(s.substr(start, pos - start));
  JsonValue result;
  result.type = JsonValue::Type::Number;
  result.number_value = value;
  return result;
}

JsonValue parse_literal(const std::string& literal, JsonValue::Type type, bool bool_value = false) {
  JsonValue v;
  v.type = type;
  v.bool_value = bool_value;
  return v;
}

JsonValue parse_value(const std::string& s, std::size_t& pos) {
  skip_ws(s, pos);
  if (pos >= s.size()) {
    throw std::runtime_error("JSON parse error: unexpected end of input");
  }
  const char ch = s[pos];
  if (ch == '{') {
    return parse_object(s, pos);
  }
  if (ch == '[') {
    return parse_array(s, pos);
  }
  if (ch == '"') {
    JsonValue v;
    v.type = JsonValue::Type::String;
    v.string_value = parse_string(s, pos);
    return v;
  }
  if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-') {
    return parse_number(s, pos);
  }
  if (s.compare(pos, 4, "true") == 0) {
    pos += 4;
    return parse_literal("true", JsonValue::Type::Bool, true);
  }
  if (s.compare(pos, 5, "false") == 0) {
    pos += 5;
    return parse_literal("false", JsonValue::Type::Bool, false);
  }
  if (s.compare(pos, 4, "null") == 0) {
    pos += 4;
    return JsonValue{};
  }
  throw std::runtime_error("JSON parse error: unexpected token");
}

const JsonValue& expect_object_field(const JsonValue& obj, const char* key) {
  auto it = obj.object.find(key);
  if (it == obj.object.end()) {
    throw std::runtime_error(std::string("INFQ manifest missing field: ") + key);
  }
  return it->second;
}

std::size_t to_size_t(const JsonValue& v) {
  if (v.type != JsonValue::Type::Number) {
    throw std::runtime_error("INFQ manifest expected numeric value");
  }
  return static_cast<std::size_t>(v.number_value + 0.5);
}

std::string to_string(const JsonValue& v) {
  if (v.type != JsonValue::Type::String) {
    throw std::runtime_error("INFQ manifest expected string value");
  }
  return v.string_value;
}

TensorInfo parse_tensor(const JsonValue& node) {
  const JsonValue& name = expect_object_field(node, "name");
  const JsonValue& dtype = expect_object_field(node, "dtype");
  const JsonValue& rows = expect_object_field(node, "rows");
  const JsonValue& cols = expect_object_field(node, "cols");
  const JsonValue& block = expect_object_field(node, "block");
  const JsonValue& scale_dtype = expect_object_field(node, "scale_dtype");
  const JsonValue& data_file = expect_object_field(node, "data_file");
  const JsonValue& offset_data = expect_object_field(node, "offset_data");
  const JsonValue& offset_scales = expect_object_field(node, "offset_scales");

  TensorInfo info;
  info.name = to_string(name);
  info.dtype = to_string(dtype);
  info.rows = to_size_t(rows);
  info.cols = to_size_t(cols);
  info.block = to_size_t(block);
  info.scale_dtype = to_string(scale_dtype);
  info.data_file = to_string(data_file);
  info.data_offset = to_size_t(offset_data);
  info.scale_offset = to_size_t(offset_scales);
  auto outliers_it = node.object.find("outliers");
  if (outliers_it != node.object.end()) {
    const JsonValue& outlier = outliers_it->second;
    if (outlier.type == JsonValue::Type::Object) {
      OutlierInfo meta;
      meta.data_file = to_string(expect_object_field(outlier, "data_file"));
      meta.offset = to_size_t(expect_object_field(outlier, "offset"));
      meta.count = to_size_t(expect_object_field(outlier, "count"));
      auto rb = outlier.object.find("record_bytes");
      if (rb != outlier.object.end()) {
        meta.record_bytes = to_size_t(rb->second);
      }
      auto align = outlier.object.find("align");
      if (align != outlier.object.end()) {
        meta.align = to_size_t(align->second);
      }
      auto layout = outlier.object.find("layout");
      if (layout != outlier.object.end()) {
        meta.layout = to_string(layout->second);
      }
      info.outliers = meta;
    }
  }
  return info;
}

}  // namespace

InfqModel::InfqModel(std::string manifest_path) {
  std::ifstream in(manifest_path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open INFQ manifest: " + manifest_path);
  }
  std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::size_t pos = 0;
  JsonValue root = parse_value(contents, pos);
  if (root.type != JsonValue::Type::Object) {
    throw std::runtime_error("INFQ manifest root must be an object");
  }
  const JsonValue& tensors = expect_object_field(root, "tensors");
  if (tensors.type != JsonValue::Type::Array) {
    throw std::runtime_error("INFQ manifest 'tensors' must be an array");
  }
  for (const auto& t : tensors.array) {
    if (t.type != JsonValue::Type::Object) {
      throw std::runtime_error("INFQ manifest tensor entry must be an object");
    }
    tensors_.push_back(parse_tensor(t));
  }

  const JsonValue& adapters = expect_object_field(root, "adapters");
  if (adapters.type != JsonValue::Type::Array) {
    throw std::runtime_error("INFQ manifest 'adapters' must be an array");
  }
  for (const auto& a : adapters.array) {
    if (a.type != JsonValue::Type::Object) {
      throw std::runtime_error("INFQ manifest adapter entry must be an object");
    }
    AdapterInfo info;
    info.name = to_string(expect_object_field(a, "name"));
    info.rank = static_cast<std::int32_t>(to_size_t(expect_object_field(a, "rank")));
    info.A = parse_tensor(expect_object_field(a, "A"));
    info.B = parse_tensor(expect_object_field(a, "B"));
    adapters_.push_back(std::move(info));
  }
}

InfqModel::~InfqModel() = default;

}  // namespace infeng::quant
