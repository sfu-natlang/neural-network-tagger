PARAMS=128-0.08-3600-0.9-0
for SET in dev; do
	bazel-bin/syntaxnet/parser_eval \
		--task_context=models/brain_pos/greedy/$PARAMS/context \
		--hidden_layer_sizes=128 \
		--input=$SET-corpus \
		--output=tagged-$SET-corpus \
		--arg_prefix=brain_pos \
		--graph_builder=greedy \
		--model_path=models/brain_pos/greedy/$PARAMS/model
done
