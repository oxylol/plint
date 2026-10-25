// Plint TV · audiodev — list / read / set the default sound output.
//   audiodev list            one output device per line
//   audiodev current         the current default output
//   audiodev set "<name>"    make that device the default output
// Built by tv/install.sh:  clang -O2 -framework CoreAudio -framework CoreFoundation -o audiodev audiodev.c
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <string.h>

#ifndef kAudioObjectPropertyElementMain
#define kAudioObjectPropertyElementMain kAudioObjectPropertyElementMaster
#endif

static int name_of(AudioDeviceID id, char *buf, size_t n) {
    CFStringRef s = NULL;
    UInt32 sz = sizeof(s);
    AudioObjectPropertyAddress a = { kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    if (AudioObjectGetPropertyData(id, &a, 0, NULL, &sz, &s) != noErr || s == NULL) return 0;
    int ok = CFStringGetCString(s, buf, (CFIndex)n, kCFStringEncodingUTF8);
    CFRelease(s);
    return ok;
}

static int has_output(AudioDeviceID id) {
    AudioObjectPropertyAddress a = { kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain };
    UInt32 sz = 0;
    if (AudioObjectGetPropertyDataSize(id, &a, 0, NULL, &sz) != noErr) return 0;
    return sz > 0;
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "list";
    AudioObjectPropertyAddress def = { kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    char name[256];

    if (strcmp(cmd, "current") == 0) {
        AudioDeviceID cur = 0;
        UInt32 csz = sizeof(cur);
        if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &def, 0, NULL, &csz, &cur) == noErr && name_of(cur, name, sizeof name)) {
            puts(name);
            return 0;
        }
        return 1;
    }

    AudioObjectPropertyAddress all = { kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    UInt32 sz = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &all, 0, NULL, &sz) != noErr || sz == 0) return 1;
    int n = (int)(sz / sizeof(AudioDeviceID));
    AudioDeviceID ids[n];
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &all, 0, NULL, &sz, ids) != noErr) return 1;

    for (int i = 0; i < n; i++) {
        if (!has_output(ids[i]) || !name_of(ids[i], name, sizeof name)) continue;
        if (strcmp(cmd, "list") == 0) {
            puts(name);
        } else if (strcmp(cmd, "set") == 0 && argc > 2 && strcmp(name, argv[2]) == 0) {
            return AudioObjectSetPropertyData(kAudioObjectSystemObject, &def, 0, NULL, sizeof(ids[i]), &ids[i]) == noErr ? 0 : 1;
        }
    }
    if (strcmp(cmd, "list") == 0) return 0;
    fprintf(stderr, "usage: audiodev list | current | set \"<name>\"\n");
    return 2;
}
