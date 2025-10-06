# Install the whole PC15 stack as ONE package

Public GitHub tag install (no token, no PyPI):

```bash
pip install "git+https://github.com/agaloppe84/ProceduralCodec-v15@v15.0.0"
```

Then:

```python
import pc15 as pc
print(pc.__version__)
print(pc.proc.list_generators()[:5])  # if GPU deps present
```
