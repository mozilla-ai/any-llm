from typing import Any


def _patch_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Patch the JSON schema to be compatible with Llama's API."""
    # Llama requires that the 'union_specified' has a parameter type specified
    # so we need to patch the schema to include the type of the parameter
    # if any of the function call parameter properties have 'oneOf' set, make sure the property has type set. If not, set it to string. This is a quirk with Llama API currently.
    props = schema["function"]["parameters"]["properties"]
    for prop in props:
        if "oneOf" in props[prop] and "type" not in props[prop]:
            props[prop]["type"] = "string"

    return schema
