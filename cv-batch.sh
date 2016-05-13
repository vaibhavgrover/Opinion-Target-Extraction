#!/bin/sh
dataset=$1
embed=$2
dim=$3
hidden=$4
model=$5

if [ "$embed" == "Google" ]; then
    file=embeddings/google-news/GoogleNews-vectors-negative300.bin.gz
elif [ "$embed" == "Amazon" ]; then
    file=embeddings/amazon/vectors-${dim}.txt
else
    file=embeddings/senna
fi

#make prepare-folds dataset=${dataset}
mkdir -p cv-result/${model} cv-${dataset}/${model}
for fold in {0..9}
do
    make prepare-json dataset=${dataset} fold=${fold} embed=${embed} EMBEDDIING_FILE=${file}
    make run-rnn dataset=${dataset} type=${model} embed=${embed} window=3 nhidden=${hidden} dimension=${dim} init=true > cv-${dataset}/${model}/${embed}-${dim}-${hidden}-${fold}.log
done

