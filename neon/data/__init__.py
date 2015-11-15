# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from neon.data.dataiterator import DataIterator
from neon.data.text import Text
from neon.data.image import Image, ImgMaster
from neon.data.speech import Speech
from neon.data.video import Video
from neon.data.loader import (load_text, load_mnist, load_cifar10, load_flickr8k, load_flickr30k,
                              load_coco, load_i1kmeta, load_places2_mini)
from neon.data.imagecaption import ImageCaption, ImageCaptionTest
