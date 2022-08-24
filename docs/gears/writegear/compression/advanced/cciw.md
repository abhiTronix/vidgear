<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
-->

# Custom FFmpeg Commands in WriteGear API

WriteGear API now provides the **[`execute_ffmpeg_cmd`](../../../../../bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.execute_ffmpeg_cmd) Method** in [Compression Mode](../../overview/) that enables the user to pass any custom FFmpeg CLI _(Command Line Interface)_ commands as input to its internal FFmpeg Pipeline by formating it as a list. 

This opens endless possibilities of exploiting every FFmpeg params within WriteGear without relying on a third-party API to do the same and while doing that it robustly handles all errors/warnings quietly.


&nbsp;


!!! warning "Important Information"

    * This Feature Requires WriteGear's [Compression Mode enabled(`compression_mode = True`)](../../params/#compression_mode). Follow these dedicated [Installation Instructions ➶](../../advanced/ffmpeg_install/) for its installation.

    * ==Only **python list** is a valid datatype as input for this function, any other value will throw `ValueError`.==

    * Kindly read [**FFmpeg Docs**](https://ffmpeg.org/documentation.html) carefully, before passing any values to `output_param` dictionary parameter. Wrong values may result in undesired Errors or no output at all.

&nbsp;

## Features 

- [x] Provides the ability to pass any custom command to WriteGear FFmpeg Pipeline.

- [x] Compatible with any FFmpeg terminal command.

- [x] Standalone On-the-fly functioning.

- [x] Can work without interfering with WriteGear API's Writer pipeline.

- [x] Minimum hassle and extremely easy to enable and use. 



&nbsp;


## Methods

### **[`execute_ffmpeg_cmd`](../../../../../bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.execute_ffmpeg_cmd)** 

This method allows the users to pass the custom FFmpeg terminal commands as a _**formatted list**_ directly to WriteGear API's FFmpeg pipeline for processing/execution. Its usage is as follows: 
  
```python
# format FFmpeg terminal command `ffmpeg -y -i source_video -acodec copy input_audio.aac` as a list
ffmpeg_command = ["-y", "-i", source_video, "-acodec", "copy", "input_audio.aac"]

# execute this list using this function
execute_ffmpeg_cmd(ffmpeg_command)
```


&nbsp;


## Usage Examples

!!! abstract "Following usage examples is just an idea of what can be done with this powerful function. So just Tinker with various FFmpeg parameters/commands yourself and see it working. Also, if you're unable to run any terminal FFmpeg command, then [report an issue](../../../../../contribution/issue/)."


### Using WriteGear to separate Audio from Video

In this example, we will extract and save audio from a URL stream:

```python hl_lines="13-18 21"
# import required libraries
from vidgear.gears import WriteGear

# define a valid url
url_to_stream = (
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
)

# Define writer with default parameters
writer = WriteGear(output="Output.mp4", logging=True)

# format command to convert stream audio as 'output_audio.aac' as list
ffmpeg_command_to_save_audio = [
    "-y",
    "-i",
    url_to_stream,
    "output_audio.aac",
]  # `-y` parameter is to overwrite outputfile if exists

# execute FFmpeg command
writer.execute_ffmpeg_cmd(ffmpeg_command_to_save_audio)

# safely close writer
writer.close()
```

After running this script, You will get the final `'output_audio.aac'` audio file.

&nbsp;

### Using WriteGear to merge Audio with Video

In this example, we will merge audio with video:

!!! tip "You can also directly add external audio input to video-frames in WriteGear. For more information, See [this FAQ example ➶](../../../../../help/writegear_faqs/#how-add-external-audio-file-input-to-video-frames)"

!!! alert "Example Assumptions"

    * You already have a separate video(i.e `'input-video.mp4'`) and audio(i.e `'input-audio.aac'`) files.

    * Both these Audio and Video files are compatible.


```python hl_lines="59-75 78"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2
import time

# Open input video stream
stream = VideoGear(source="input-video.mp4").start()

# set input audio stream path
input_audio = "input-audio.aac"

# define your parameters
output_params = {
    "-input_framerate": stream.framerate
}  # output framerate must match source framerate

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4", **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()

# safely close writer
writer.close()


# sleep 1 sec as the above video might still be rendering
time.sleep(1)


# format FFmpeg command to generate `Output_with_audio.mp4` by merging input_audio in above rendered `Output.mp4`
ffmpeg_command = [
    "-y",
    "-i",
    "Output.mp4",
    "-i",
    input_audio,
    "-c:v",
    "copy",
    "-c:a",
    "copy",
    "-map",
    "0:v:0",
    "-map",
    "1:a:0",
    "-shortest",
    "Output_with_audio.mp4",
]  # `-y` parameter is to overwrite outputfile if exists

# execute FFmpeg command
writer.execute_ffmpeg_cmd(ffmpeg_command)
```

After running this script, You will get the final `'Output_with_audio.mp4'` file with both video and audio merged.

&nbsp;