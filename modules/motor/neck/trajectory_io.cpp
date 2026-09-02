#include "neck/trajectory_io.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

namespace {

struct JsonValue {
    enum class Type { Null, Boolean, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    bool parse(JsonValue& output, std::string& error) {
        skipWhitespace();
        if (!parseValue(output)) {
            error = error_;
            return false;
        }
        skipWhitespace();
        if (position_ != text_.size()) {
            fail("unexpected characters after JSON value");
            error = error_;
            return false;
        }
        return true;
    }

private:
    void skipWhitespace() {
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                break;
            }
            ++position_;
        }
    }

    bool fail(const std::string& message) {
        if (error_.empty()) {
            error_ = message + " at byte " + std::to_string(position_);
        }
        return false;
    }

    bool parseValue(JsonValue& output) {
        skipWhitespace();
        if (position_ >= text_.size()) {
            return fail("unexpected end of JSON");
        }

        switch (text_[position_]) {
            case '{':
                return parseObject(output);
            case '[':
                return parseArray(output);
            case '"':
                output.type = JsonValue::Type::String;
                return parseString(output.string);
            case 't':
                output.type = JsonValue::Type::Boolean;
                output.boolean = true;
                return parseLiteral("true");
            case 'f':
                output.type = JsonValue::Type::Boolean;
                output.boolean = false;
                return parseLiteral("false");
            case 'n':
                output.type = JsonValue::Type::Null;
                return parseLiteral("null");
            default:
                if (text_[position_] == '-' ||
                    (text_[position_] >= '0' && text_[position_] <= '9')) {
                    return parseNumber(output);
                }
                return fail("unexpected JSON token");
        }
    }

    bool parseLiteral(const char* literal) {
        const std::string expected(literal);
        if (text_.compare(position_, expected.size(), expected) != 0) {
            return fail("invalid JSON literal");
        }
        position_ += expected.size();
        return true;
    }

    bool parseObject(JsonValue& output) {
        output.type = JsonValue::Type::Object;
        output.object.clear();
        ++position_;
        skipWhitespace();
        if (position_ < text_.size() && text_[position_] == '}') {
            ++position_;
            return true;
        }

        while (true) {
            skipWhitespace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                return fail("object key must be a string");
            }
            std::string key;
            if (!parseString(key)) {
                return false;
            }
            skipWhitespace();
            if (position_ >= text_.size() || text_[position_] != ':') {
                return fail("expected ':' after object key");
            }
            ++position_;

            JsonValue value;
            if (!parseValue(value)) {
                return false;
            }
            if (!output.object.emplace(key, std::move(value)).second) {
                return fail("duplicate object key '" + key + "'");
            }

            skipWhitespace();
            if (position_ >= text_.size()) {
                return fail("unterminated object");
            }
            if (text_[position_] == '}') {
                ++position_;
                return true;
            }
            if (text_[position_] != ',') {
                return fail("expected ',' or '}' in object");
            }
            ++position_;
        }
    }

    bool parseArray(JsonValue& output) {
        output.type = JsonValue::Type::Array;
        output.array.clear();
        ++position_;
        skipWhitespace();
        if (position_ < text_.size() && text_[position_] == ']') {
            ++position_;
            return true;
        }

        while (true) {
            JsonValue value;
            if (!parseValue(value)) {
                return false;
            }
            output.array.push_back(std::move(value));

            skipWhitespace();
            if (position_ >= text_.size()) {
                return fail("unterminated array");
            }
            if (text_[position_] == ']') {
                ++position_;
                return true;
            }
            if (text_[position_] != ',') {
                return fail("expected ',' or ']' in array");
            }
            ++position_;
        }
    }

    static void appendUtf8(unsigned code_point, std::string& output) {
        if (code_point <= 0x7F) {
            output.push_back(static_cast<char>(code_point));
        } else if (code_point <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
            output.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        } else if (code_point <= 0xFFFF) {
            output.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
            output.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        } else {
            output.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
            output.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        }
    }

    bool parseHex4(unsigned& value) {
        if (position_ + 4 > text_.size()) {
            return fail("incomplete Unicode escape");
        }
        value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value += static_cast<unsigned>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value += static_cast<unsigned>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value += static_cast<unsigned>(c - 'A' + 10);
            } else {
                return fail("invalid Unicode escape");
            }
        }
        return true;
    }

    bool parseString(std::string& output) {
        ++position_;  // opening quote
        output.clear();
        while (position_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') {
                return true;
            }
            if (c < 0x20) {
                return fail("unescaped control character in string");
            }
            if (c != '\\') {
                output.push_back(static_cast<char>(c));
                continue;
            }
            if (position_ >= text_.size()) {
                return fail("incomplete string escape");
            }
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    unsigned code_point = 0;
                    if (!parseHex4(code_point)) {
                        return false;
                    }
                    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
                        if (position_ + 2 > text_.size() ||
                            text_[position_] != '\\' || text_[position_ + 1] != 'u') {
                            return fail("high surrogate without low surrogate");
                        }
                        position_ += 2;
                        unsigned low = 0;
                        if (!parseHex4(low) || low < 0xDC00 || low > 0xDFFF) {
                            return fail("invalid low surrogate");
                        }
                        code_point = 0x10000 +
                            ((code_point - 0xD800) << 10) + (low - 0xDC00);
                    } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
                        return fail("unexpected low surrogate");
                    }
                    appendUtf8(code_point, output);
                    break;
                }
                default:
                    return fail("invalid string escape");
            }
        }
        return fail("unterminated string");
    }

    bool parseNumber(JsonValue& output) {
        const std::size_t start = position_;
        if (text_[position_] == '-') {
            ++position_;
        }
        if (position_ >= text_.size()) {
            return fail("incomplete number");
        }
        if (text_[position_] == '0') {
            ++position_;
            if (position_ < text_.size() && text_[position_] >= '0' &&
                text_[position_] <= '9') {
                return fail("leading zero in number");
            }
        } else if (text_[position_] >= '1' && text_[position_] <= '9') {
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
        } else {
            return fail("number requires an integer part");
        }

        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const std::size_t fraction_start = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
            if (position_ == fraction_start) {
                return fail("number requires digits after decimal point");
            }
        }
        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            const std::size_t exponent_start = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
            if (position_ == exponent_start) {
                return fail("number requires exponent digits");
            }
        }

        const std::string token = text_.substr(start, position_ - start);
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0' || !std::isfinite(value)) {
            return fail("number is not finite");
        }
        output.type = JsonValue::Type::Number;
        output.number = value;
        return true;
    }

    const std::string& text_;
    std::size_t position_ = 0;
    std::string error_;
};

const JsonValue* member(const JsonValue& object, const std::string& key) {
    const auto it = object.object.find(key);
    return it == object.object.end() ? nullptr : &it->second;
}

bool requireMember(const JsonValue& object, const std::string& key,
                   JsonValue::Type type, const JsonValue*& output,
                   std::string& error) {
    output = member(object, key);
    if (output == nullptr) {
        error = "missing required field '" + key + "'";
        return false;
    }
    if (output->type != type) {
        error = "field '" + key + "' has the wrong JSON type";
        return false;
    }
    return true;
}

bool parseState(const std::string& text, BehaviorState& state) {
    if (text == "speaking") {
        state = BehaviorState::Speaking;
        return true;
    }
    if (text == "listening") {
        state = BehaviorState::Listening;
        return true;
    }
    if (text == "silent") {
        state = BehaviorState::Silent;
        return true;
    }
    return false;
}

}  // namespace

bool parseTrajectoryJson(const std::string& json_text,
                         Trajectory& trajectory,
                         std::string& error_message) {
    error_message.clear();
    JsonValue root;
    JsonParser parser(json_text);
    if (!parser.parse(root, error_message)) {
        return false;
    }
    if (root.type != JsonValue::Type::Object) {
        error_message = "trajectory JSON root must be an object";
        return false;
    }

    static const std::map<std::string, bool> supported_fields = {
        {"name", true}, {"fps", true}, {"unit", true}, {"order", true},
        {"trajectory", true}, {"states", true}
    };
    for (const auto& entry : root.object) {
        if (supported_fields.find(entry.first) == supported_fields.end()) {
            error_message = "unknown trajectory field '" + entry.first + "'";
            return false;
        }
    }

    const JsonValue* name = nullptr;
    const JsonValue* fps = nullptr;
    const JsonValue* unit = nullptr;
    const JsonValue* order = nullptr;
    const JsonValue* points = nullptr;
    if (!requireMember(root, "name", JsonValue::Type::String, name, error_message) ||
        !requireMember(root, "fps", JsonValue::Type::Number, fps, error_message) ||
        !requireMember(root, "unit", JsonValue::Type::String, unit, error_message) ||
        !requireMember(root, "order", JsonValue::Type::Array, order, error_message) ||
        !requireMember(root, "trajectory", JsonValue::Type::Array, points, error_message)) {
        return false;
    }

    if (name->string.empty()) {
        error_message = "field 'name' must not be empty";
        return false;
    }
    if (!(fps->number > 0.0)) {
        error_message = "field 'fps' must be greater than 0";
        return false;
    }
    if (unit->string != "radian") {
        error_message = "field 'unit' must be 'radian'";
        return false;
    }
    if (order->array.size() != 3 ||
        order->array[0].type != JsonValue::Type::String ||
        order->array[1].type != JsonValue::Type::String ||
        order->array[2].type != JsonValue::Type::String ||
        order->array[0].string != "roll" ||
        order->array[1].string != "pitch" ||
        order->array[2].string != "yaw") {
        error_message = "field 'order' must be [\"roll\", \"pitch\", \"yaw\"]";
        return false;
    }
    if (points->array.empty()) {
        error_message = "field 'trajectory' must not be empty";
        return false;
    }

    constexpr double radians_to_degrees =
        57.295779513082320876798154814105;
    Trajectory loaded;
    loaded.name = name->string;
    loaded.fps = fps->number;
    loaded.points.reserve(points->array.size());
    for (std::size_t frame_index = 0; frame_index < points->array.size();
         ++frame_index) {
        const JsonValue& frame = points->array[frame_index];
        if (frame.type != JsonValue::Type::Array || frame.array.size() != 3) {
            error_message = "trajectory frame " + std::to_string(frame_index) +
                            " must be an array of exactly 3 numbers";
            return false;
        }
        for (const JsonValue& component : frame.array) {
            if (component.type != JsonValue::Type::Number ||
                !std::isfinite(component.number)) {
                error_message = "trajectory frame " +
                                std::to_string(frame_index) +
                                " contains a non-finite or non-numeric value";
                return false;
            }
        }
        loaded.points.push_back({
            frame.array[1].number * radians_to_degrees,
            frame.array[0].number * radians_to_degrees,
            frame.array[2].number * radians_to_degrees
        });
    }

    const JsonValue* states = member(root, "states");
    if (states != nullptr) {
        if (states->type != JsonValue::Type::Array) {
            error_message = "optional field 'states' must be an array";
            return false;
        }
        if (states->array.size() != loaded.points.size()) {
            error_message = "field 'states' must have the same length as 'trajectory'";
            return false;
        }
        loaded.states.reserve(states->array.size());
        for (std::size_t index = 0; index < states->array.size(); ++index) {
            const JsonValue& value = states->array[index];
            BehaviorState state = BehaviorState::Silent;
            if (value.type != JsonValue::Type::String ||
                !parseState(value.string, state)) {
                error_message = "states[" + std::to_string(index) +
                                "] must be speaking, listening, or silent";
                return false;
            }
            loaded.states.push_back(state);
        }
    }

    trajectory = std::move(loaded);
    return true;
}

bool loadTrajectoryJson(const std::string& path,
                        Trajectory& trajectory,
                        std::string& error_message) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error_message = "cannot open trajectory file: " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error_message = "failed while reading trajectory file: " + path;
        return false;
    }
    return parseTrajectoryJson(buffer.str(), trajectory, error_message);
}
