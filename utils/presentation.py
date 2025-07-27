import pandas as pd

BASE_TABLE_STYLES = [
    {
        'selector': 'caption',
        'props': [
            ('caption-side', 'top'),
            ('text-align', 'left'),
            ('font-size', '1.1em'),
            ('font-weight', 'bold'),
        ]
    }
]

def style_panel(df: pd.DataFrame, title: str):
    """
    Apply the common caption + 2‑decimal formatting to any slice.
    """
    return (
        df.style
        .set_caption(title)
        .set_table_styles(BASE_TABLE_STYLES)
        .format("{:.2f}")
    )

