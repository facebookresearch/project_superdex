# Third-party software

This plugin compiles the components listed below from source into its `TeleopNodeZenoh` module.
Their sources are not part of the plugin; the build downloads them (see the plugin README). Their
license texts are provided in the adjacent directories.

| Component | License | Source |
| --- | --- | --- |
| zenoh-pico 1.10.1 | EPL-2.0 OR Apache-2.0 | https://github.com/eclipse-zenoh/zenoh-pico |
| nanopb 0.4.9.1 (C runtime) | zlib | https://github.com/nanopb/nanopb |

`ThirdParty/zenoh-pico/patches` holds changes to zenoh-pico that the build applies to the
downloaded sources. `Source/TeleopNodeZenoh/Private/TeleopMsgs/generated/` holds code that the
nanopb generator produced from `teleop.proto`.
