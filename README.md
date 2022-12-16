This is the code used to create results for paper *Classification with Costly Features in Hierarchical Deep Sets*.

Directory structured as follows:
- `code`: contains the code and datasets
- `vis_tool`: javascript tool to visualize the agent's behavior
- `experiments`: scripts to perform the experiments and produce the plots

In the code folder, there are three algorithms *rl* (the main agent), *rw* (random sampling), and *mil* (the hmil agent with all information) in their corresponding directories. The `data` folder contains preprocessed datasets.

In each directory, the main.py is the main script.
Run `agent_rl` as:
`python main.py [dataset] [target lambda]`
i.e. `python main.py recipes 0.001`

Run `agent_rw` as:
`python main.py [dataset] [target budget]`
i.e. `python main.py recipes 5.0`

Run `agent_mil` as:
`python main.py [dataset]`
i.e. `python main.py recipes`

Note that `toy_b` dataset needs to be run with an additional argument `-dataseed 0`.

With a trained rl model, you can create a json for visualization in the `vis_tool` with `agent_rl/eval_vis.py` tool.
Run as:
`python eval_vis.py [dataset] -model [model file]`

The code requires numpy, pytorch and sklearn libraries.
