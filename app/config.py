class APISettings:
    
    def __init__(self) -> None:
        pass

    ########################     Global information    ########################
    
    title: str = "HomerBot"
    contacts: str = "urgellbapt@cy-tech.fr, dingleyjes@cy-tech.fr, maillotvic@cy-tech.fr"

    #docs_url: str = "/docs"
    #redoc_url: str = "/redoc"
    
    ########################        huggingface        ########################
    
    api_url: dict = {"Homer": "https://huggingface.co/DingleyMaillotUrgell/homer-bot"}
    api_token: str = "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"
    

def get_api_settings() -> APISettings:
    """Init and return the API settings

    Returns:
        APISettings: The settings
    """
    return APISettings()  # reads variables from environment