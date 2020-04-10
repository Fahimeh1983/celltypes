N=1
length=10000
p=1
q=1
walk_filename="walk_0.csv"
roi="VISp"
project_name="NPP_GNN_project"
layer_class="single_layer"
layer="base_unnormalized_allcombined"
walk_type="Directed_Weighted_node2vec"
window=2
batch_size=2000
num_workers=4
embedding_size=2
learning_rate=0.001
n_epochs=10

source activate py374

python -m trainer --N $N --length $length --p $p --q $q --walk_filename $walk_filename --roi $roi --project_name $project_name \
--layer_class $layer_class --layer $layer --walk_type $walk_type --window $window --batch_size $batch_size \
--num_workers $num_workers --embedding_size $embedding_size --learning_rate $learning_rate --n_epochs $n_epochs
