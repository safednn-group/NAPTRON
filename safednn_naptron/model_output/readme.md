## Postprocessing of model's outputs
If you need to customize postprocessing of model's outputs do the following:

1. Derive your custom output handler from OutputHandler class.
2. Add the postprocessing logic in *_process* method of your new class.
3. Add key-value pair in OUTPUT_HANDLERS dict in builder.py. 
4. Declare that you want to use the handler in the config file you use for your experiments like this:
```
output_handler = dict(
    type="simple_dump"
)
```

