for loop in 0 1 2 3 4
do
    for dataset in 'korea_covid'
    do
        for seq_day in 4
        do
            for pred_day in 7 14 30
            do
                python3 main.py\
                --baseline 1\
                --model_name 'DLinear'\
                --gpu_id 0\
                --batch_size 8\
                --dataset $dataset\
                --seq_day $seq_day\
                --pred_day $pred_day\
                | tee -a ./shell_logs/DLinear_${dataset}_seq${seq_day}_pred${pred_day}_loop${loop}.log
            done
        done
    done
done