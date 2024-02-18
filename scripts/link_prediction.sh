# # ogbl-collab
python main.py  --type link_prediction \
                 --data ogbl-collab \
                 --model GMT \
                 --model-string GMPool_G-SelfAtt-GMPool_I \
                 --gpu $1 \
                 --experiment-number $2 \
                 --batch-size 1000 \
                 --num-hidden 128 \
                 --num-heads 1 \
                 --lr-schedule \
                 --cluster