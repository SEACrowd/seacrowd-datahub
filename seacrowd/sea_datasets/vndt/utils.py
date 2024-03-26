import typing as T

from conllu.exceptions import ParseException
from conllu.models import Metadata, TokenList
from conllu.parser import (DEFAULT_FIELD_PARSERS, DEFAULT_FIELDS,
                           _FieldParserType, _MetadataParserType,
                           parse_comment_line, parse_line)

imputed_sent_id: int = 1


def parse_token_and_impute_metadata(data: str, fields: T.Optional[T.Sequence[str]] = None, field_parsers: T.Optional[T.Dict[str, _FieldParserType]] = None, metadata_parsers: T.Optional[T.Dict[str, _MetadataParserType]] = None) -> TokenList:
    """
    Overrides conllu.parse_token_and_metadata via monkey patching.
    This function imputes the following metadata, if these are not found in the .conllu file:
        - sent_id (int): an integer identifier for each sentence.
        - text (str): a concatenated string of token forms. This assumes that all token forms
        are separated with an empty space ' ', and does not consider the `SpaceAfter` field.
    """

    if not data:
        raise ParseException("Can't create TokenList, no data sent to constructor.")

    fields = fields or DEFAULT_FIELDS
    global imputed_sent_id

    if not field_parsers:
        field_parsers = DEFAULT_FIELD_PARSERS.copy()
    elif sorted(field_parsers.keys()) != sorted(fields):
        new_field_parsers = DEFAULT_FIELD_PARSERS.copy()
        new_field_parsers.update(field_parsers)
        field_parsers = new_field_parsers

    tokens = []
    metadata = Metadata()

    for line in data.split('\n'):
        line = line.strip()

        if not line:
            continue

        if line.startswith('#'):
            pairs = parse_comment_line(line, metadata_parsers=metadata_parsers)
            for key, value in pairs:
                metadata[key] = value

        else:
            tokens.append(parse_line(line, fields, field_parsers))

    if 'sent_id' not in metadata:
        metadata['sent_id'] = str(imputed_sent_id)
        imputed_sent_id += 1

    if 'text' not in metadata:
        imputed_text = ""
        for token in tokens:
            imputed_text += str(token['form']) + " "
        metadata['text'] = imputed_text

    return TokenList(tokens, metadata, default_fields=fields)
