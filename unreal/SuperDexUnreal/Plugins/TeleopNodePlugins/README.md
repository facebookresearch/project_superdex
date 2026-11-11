# Teleop node plugins

Optional plugins that carry teleop node traffic (`UTeleopNodeBaseComponent`, SuperDexTeleop module of
SuperDexRobotics) between an operator and a robot. Each backend is a `UTeleopNodeImplBase` subclass in
its own plugin, so SuperDexRobotics does not depend on any transport: enable only the backends your
project uses.

| Plugin | Backend | Needs |
| --- | --- | --- |
| TeleopNodeZenoh (Win64, Android) | `UTeleopNodeImplZenoh`: zenoh through zenoh-pico, with protobuf messages encoded by nanopb | zenoh-pico and nanopb sources in the plugin (see its README) |

## How a backend is chosen

- **Per node.** A TeleopNode's `CommunicationClassName` names the backend class, for example
  `/Script/TeleopNodeZenoh.TeleopNodeImplZenoh`.
- **Default.** A node with an empty `CommunicationClassName` uses the project's **Default
  Communication Class Name** (Project Settings › Plugins › SuperDex Teleop, a dropdown of the loaded
  backends, stored in `Config/DefaultGame.ini`), or `UTeleopNodeImplDirect` (in-process, no network)
  if the project sets none.
