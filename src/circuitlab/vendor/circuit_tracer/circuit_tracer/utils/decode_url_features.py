import json
import urllib.parse
from collections import namedtuple

Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])


def decode_url_features(url: str) -> tuple[dict[str, list[Feature]], list[Feature]]:
    """
    Extract both supernode features and individual singleton features from URL.

    Returns:
        Tuple of (supernode_features, singleton_features)
        - supernode_features: Dict mapping supernode names to lists of Features
        - singleton_features: List of individual Feature objects
    """
    decoded = urllib.parse.unquote(url)

    parsed_url = urllib.parse.urlparse(decoded)
    query_params = urllib.parse.parse_qs(parsed_url.query)

    # Extract supernodes
    supernodes_json = query_params.get("supernodes", ["[]"])[0]
    supernodes_data = json.loads(supernodes_json)

    supernode_features = {}
    name_counts = {}

    for supernode in supernodes_data:
        name = supernode[0]
        node_ids = supernode[1:]

        # Handle duplicate names by adding counter
        if name in name_counts:
            name_counts[name] += 1
            unique_name = f"{name} ({name_counts[name]})"
        else:
            name_counts[name] = 1
            unique_name = name

        nodes = []
        for node_id in node_ids:
            layer, feature_idx, pos = map(int, node_id.split("_"))
            nodes.append(Feature(layer, pos, feature_idx))

        supernode_features[unique_name] = nodes

    # Extract individual/singleton features from pinnedIds
    pinned_ids_str = query_params.get("pinnedIds", [""])[0]
    singleton_features = []

    if pinned_ids_str:
        pinned_ids = pinned_ids_str.split(",")
        for pinned_id in pinned_ids:
            # Handle both regular format (layer_feature_pos) and E_ format
            if pinned_id.startswith("E_"):
                # E_26865_9 format - embedding layer
                parts = pinned_id[2:].split("_")  # Remove 'E_' prefix
                if len(parts) == 2:
                    feature_idx, pos = map(int, parts)
                    # Use -1 to indicate embedding layer
                    singleton_features.append(Feature(-1, pos, feature_idx))
            else:
                # Regular layer_feature_pos format
                parts = pinned_id.split("_")
                if len(parts) == 3:
                    layer, feature_idx, pos = map(int, parts)
                    singleton_features.append(Feature(layer, pos, feature_idx))

    return supernode_features, singleton_features
