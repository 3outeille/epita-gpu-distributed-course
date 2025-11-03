- `git clone https://github.com/huggingface/picotron_tutorial/tree/cpu`
- Setup `uv env`
    ```
    uv venv dummy_env --python 3.12
    source dummy_env/bin/activate
    uv pip install torch numpy datasets transformers lovely_tensors wandb debugpy-run
    ```
- Setup [debugger in Vscode](https://www.youtube.com/watch?v=_8xlRgFY_-g)

# Ressources

- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/3outeille/picotron_tutorial
- https://huggingface.co/spaces/nanotron/ultrascale-playbook
- https://jax-ml.github.io/scaling-book/
