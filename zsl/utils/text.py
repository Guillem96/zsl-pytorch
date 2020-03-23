import string
import functools

import wikipedia


@functools.lru_cache(maxsize=200)
def wikipedia_description(search_term: str, 
                          remove_punct: bool = False,
                          to_lower: bool = False,
                          **kwargs) -> str:
    """
    Gets a description from wikipedia given a search term

    This method adds extra functionality to `wikipedia.summary` function.
    To learn more about it check: https://github.com/goldsmith/Wikipedia

    Parameters
    ----------
    search_term: str
        Word or set of words to look for at wikipedia.
    remove_punct: bool, default False
        If set to true, the punctuation of the resulting description will be 
        removed
    to_lower: bool, default False
        If set to true, the description will be transformed to lowercase
    kwargs: delegated parameters to `wikipedia.summary`

    Returns
    -------
    str
        Wikipedia description
    """
    summary = wikipedia.summary(search_term, **kwargs)
    
    if to_lower:
        summary = summary.lower()
    
    if remove_punct:
        punct_set = set(string.punctuation)
        summary = ''.join(' ' if o in punct_set else o for o in summary)
    
    return summary
