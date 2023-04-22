import os
import re
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import whisper
from datasets import load_dataset
from jiwer import wer
from TTS.api import TTS
