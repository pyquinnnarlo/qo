(venv) fyplg@FYPLG:~/Desktop/pi/exam $ sudo apt update
sudo apt install python3-opencv -y
Hit:1 http://archive.raspberrypi.com/debian trixie InRelease
Hit:2 http://deb.debian.org/debian trixie InRelease
Hit:3 http://deb.debian.org/debian trixie-updates InRelease
Hit:4 http://deb.debian.org/debian-security trixie-security InRelease
4 packages can be upgraded. Run 'apt list --upgradable' to see them.
python3-opencv is already the newest version (4.10.0+dfsg-5).
Summary:
  Upgrading: 0, Installing: 0, Removing: 0, Not Upgrading: 4
(venv) fyplg@FYPLG:~/Desktop/pi/exam $ python main.py 
 [DB] Loading Student Faces...
   + Loaded: john_doe
 [DB] System Ready. Known students: 1
 [VISION] Attempting to start camera via GStreamer...
 * Serving Flask app 'main'
 * Debug mode: off
[1:18:05.175242585] [4133]  INFO Camera camera_manager.cpp:340 libcamera v0.6.0+rpt20251202
[1:18:05.184109039] [4155]  INFO RPI pisp.cpp:720 libpisp version 1.3.0
[1:18:05.187339778] [4155]  INFO IPAProxy ipa_proxy.cpp:180 Using tuning file /usr/share/libcamera/ipa/rpi/pisp/ov5647.json
[1:18:05.194092950] [4155]  INFO Camera camera_manager.cpp:223 Adding camera '/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36' for pipeline handler rpi/pisp
[1:18:05.194132674] [4155]  INFO RPI pisp.cpp:1181 Registered camera /base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36 to CFE device /dev/media2 and ISP device /dev/media0 using PiSP variant BCM2712_D0
[1:18:05.197666088] [4158]  INFO Camera camera.cpp:1215 configuring streams: (0) 640x480-SGBRG16/RAW
[1:18:05.197774925] [4155]  INFO RPI pisp.cpp:1485 Sensor: /base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36 - Selected sensor format: 640x480-SGBRG10_1X10/RAW - Selected CFE format: 640x480-GB16/RAW
[ WARN:0@8.783] global cap_gstreamer.cpp:2839 handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module libcamerasrc0 reported: Internal data stream error.
[ WARN:0@8.784] global cap_gstreamer.cpp:1698 open OpenCV | GStreamer warning: unable to start pipeline
[ WARN:0@8.785] global cap_gstreamer.cpp:1173 isPipelinePlaying OpenCV | GStreamer warning: GStreamer: pipeline have not been created
[ WARN:0@8.786] global cap.cpp:204 open VIDEOIO(GSTREAMER): backend is generally available but can't be used to capture by name
 [ERROR] Failed to open camera! Check if 'gstreamer1.0-libcamera' is installed.


