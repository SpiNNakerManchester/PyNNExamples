import spynnaker8 as p
import sys

if len(sys.argv) < 3:
    print("pushbot_retina_viewer.py <port> <resolution>")
    sys.exit(1)

port = int(sys.argv[1])
retina_resolution = p.external_devices.PushBotRetinaResolution[sys.argv[2]]

viewer = p.external_devices.PushBotRetinaViewer(
    retina_resolution.value, port=port)
print(viewer.local_port)
viewer.run()
