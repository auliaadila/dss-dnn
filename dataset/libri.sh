./download_data.sh
python libri.py LibriSpeech Libri1h --train-minutes 60.0 --val-minutes 4.0 --test-minutes 4.0 --format wav
# python libri.py LibriSpeech Libri2h --test-minutes 8.0 --format wav