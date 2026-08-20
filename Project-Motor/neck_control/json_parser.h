// 最小 JSON 解析器（自包含，无第三方依赖）。
// 支持对象/数组/字符串/数字/布尔/null，用于解析上游轨迹 JSON。
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace neck_control {
namespace json {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolVal = false;
    double numVal = 0.0;
    std::string strVal;
    std::vector<ValuePtr> arr;
    std::vector<std::pair<std::string, ValuePtr>> obj; // 保持插入顺序

    bool isNull() const { return type == Type::Null; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray() const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    // 便捷访问
    const ValuePtr find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : obj)
            if (kv.first == key) return kv.second;
        return nullptr;
    }
    size_t size() const {
        if (type == Type::Array) return arr.size();
        if (type == Type::Object) return obj.size();
        return 0;
    }
};

// 解析 JSON 文本。成功返回非空 Value；失败返回 nullptr 并写入 err。
ValuePtr parse(const std::string& text, std::string* err = nullptr);

} // namespace json
} // namespace neck_control
