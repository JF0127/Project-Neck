#include "neck/trajectory.h"

const char* behaviorStateString(BehaviorState state) {
    switch (state) {
        case BehaviorState::Speaking:
            return "speaking";
        case BehaviorState::Listening:
            return "listening";
        case BehaviorState::Silent:
            return "silent";
    }
    return "unknown";
}
