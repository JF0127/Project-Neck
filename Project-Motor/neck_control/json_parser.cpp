#include "json_parser.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace neck_control {
namespace json {

namespace {

struct Parser {
    const std::string& s;
    size_t pos = 0;
    std::string err;

    explicit Parser(const std::string& text) : s(text) {}

    void skipWs() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
    }

    bool fail(const std::string& msg) {
        if (err.empty()) err = msg + " (位置 " + std::to_string(pos) + ")";
        return false;
    }

    bool expect(char c) {
        skipWs();
        if (pos < s.size() && s[pos] == c) { ++pos; return true; }
        return fail(std::string("期望 '") + c + "'");
    }

    ValuePtr parseValue() {
        skipWs();
        if (pos >= s.size()) { fail("意外结束"); return nullptr; }
        char c = s[pos];
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't') return parseLiteral("true", true);
        if (c == 'f') return parseLiteral("false", false);
        if (c == 'n') return parseLiteral("null", false, true);
        if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
        fail("意外的字符");
        return nullptr;
    }

    ValuePtr parseLiteral(const char* lit, bool bval, bool isNull = false) {
        size_t len = std::strlen(lit);
        if (s.compare(pos, len, lit) != 0) { fail("非法字面量"); return nullptr; }
        pos += len;
        auto v = std::make_shared<Value>();
        v->type = isNull ? Value::Type::Null : Value::Type::Bool;
        v->boolVal = bval;
        return v;
    }

    ValuePtr parseNumber() {
        size_t start = pos;
        if (pos < s.size() && s[pos] == '-') ++pos;
        while (pos < s.size() && std::isdigit((unsigned char)s[pos])) ++pos;
        if (pos < s.size() && s[pos] == '.') {
            ++pos;
            while (pos < s.size() && std::isdigit((unsigned char)s[pos])) ++pos;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            ++pos;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
            while (pos < s.size() && std::isdigit((unsigned char)s[pos])) ++pos;
        }
        std::string num = s.substr(start, pos - start);
        if (num.empty() || num == "-") { fail("非法数字"); return nullptr; }
        char* end = nullptr;
        double d = std::strtod(num.c_str(), &end);
        if (end == nullptr || *end != '\0' || !std::isfinite(d)) { fail("数字无法解析"); return nullptr; }
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Number;
        v->numVal = d;
        return v;
    }

    ValuePtr parseString() {
        if (!expect('"')) return nullptr;
        std::string out;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') {
                auto v = std::make_shared<Value>();
                v->type = Value::Type::String;
                v->strVal = out;
                return v;
            }
            if (c == '\\') {
                if (pos >= s.size()) { fail("转义截断"); return nullptr; }
                char e = s[pos++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (pos + 4 > s.size()) { fail("\\u 截断"); return nullptr; }
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = s[pos++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= (unsigned)(h - 'A' + 10);
                            else { fail("非法 \\u 转义"); return nullptr; }
                        }
                        // 仅处理 BMP；代理对简化为替换字符（上游轨迹 JSON 不含这些）
                        if (code < 0x80) out += (char)code;
                        else if (code < 0x800) {
                            out += (char)(0xC0 | (code >> 6));
                            out += (char)(0x80 | (code & 0x3F));
                        } else {
                            out += (char)(0xE0 | (code >> 12));
                            out += (char)(0x80 | ((code >> 6) & 0x3F));
                            out += (char)(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: fail("非法转义"); return nullptr;
                }
            } else {
                out += c;
            }
        }
        fail("字符串未闭合");
        return nullptr;
    }

    ValuePtr parseArray() {
        if (!expect('[')) return nullptr;
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Array;
        skipWs();
        if (pos < s.size() && s[pos] == ']') { ++pos; return v; }
        while (true) {
            ValuePtr item = parseValue();
            if (!item) return nullptr;
            v->arr.push_back(item);
            skipWs();
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            if (pos < s.size() && s[pos] == ']') { ++pos; return v; }
            fail("数组缺少 ',' 或 ']'");
            return nullptr;
        }
    }

    ValuePtr parseObject() {
        if (!expect('{')) return nullptr;
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Object;
        skipWs();
        if (pos < s.size() && s[pos] == '}') { ++pos; return v; }
        while (true) {
            skipWs();
            if (pos >= s.size() || s[pos] != '"') { fail("对象键必须是字符串"); return nullptr; }
            ValuePtr key = parseString();
            if (!key) return nullptr;
            if (!expect(':')) return nullptr;
            ValuePtr val = parseValue();
            if (!val) return nullptr;
            v->obj.emplace_back(key->strVal, val);
            skipWs();
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            if (pos < s.size() && s[pos] == '}') { ++pos; return v; }
            fail("对象缺少 ',' 或 '}'");
            return nullptr;
        }
    }
};

} // namespace

ValuePtr parse(const std::string& text, std::string* err) {
    Parser p(text);
    ValuePtr v = p.parseValue();
    if (!v) {
        if (err) *err = p.err.empty() ? "解析失败" : p.err;
        return nullptr;
    }
    p.skipWs();
    if (p.pos != text.size()) {
        if (err) *err = "JSON 末尾有多余内容 (位置 " + std::to_string(p.pos) + ")";
        return nullptr;
    }
    return v;
}

} // namespace json
} // namespace neck_control
