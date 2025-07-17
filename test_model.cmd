call C:\Users\bxi\miniconda3\Scripts\activate.bat
call conda activate arm
cd C:\Users\bxi\lerobot
rmdir /s /q C:\Users\bxi\.cache\huggingface\lerobot\MoeOver\eval_record-test
python -m lerobot.record --robot.type=bxi_arm --robot.id=my_awesome_follower_arm --robot.cameras="{ wrist_camera: {type: opencv, index_or_path: 0, width: 640 ,height: 480, fps: 30},overhead_camera: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --display_data=true --dataset.repo_id=MoeOver/eval_record-test --dataset.single_task="Grab the black cube" --policy.path=C:\Users\bxi\Desktop\fsdownload\pretrained_model --dataset.episode_time_s=1000

pause
