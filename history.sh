export WORKSPACE=../data/AudioSet/
time python3 utils/dataset.py download_wavs --csv_path=$WORKSPACE"/metadata/eval_segments.csv" --audios_dir=$WORKSPACE"/audios/eval_segments"
python3 utils/dataset.py pack_waveforms_to_hdf5 --audios_dir $WORKSPACE/audios/balanced_train_segments --waveforms_hdf5_path $WORKSPACE/hdf5s/waveforms/balanced_train.h5 --csv_path $WORKSPACE/metadata/balanced_train_segments.csv --no_prepend_y
python3 utils/dataset.py pack_waveforms_to_hdf5 --audios_dir $WORKSPACE/audios/eval_segments --waveforms_hdf5_path $WORKSPACE/hdf5s/waveforms/eval.h5 --csv_path $WORKSPACE/metadata/eval_segments.csv
python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/balanced_train.h5"
python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/eval.h5"
python3  pytorch/main.py train --workspace="$WORKSPACE" --data_type='balanced_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='MobileNetV2' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='balanced_train' --sample_rate=16000 --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000 --model_type='MobileNetV2' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000
