def build_context(docs, max_chars: int = 3500) -> str:
    """
    Build context by preserving reranked order
    but truncating by total character length instead of doc count.
    This preserves factual coverage.
    """
    context_parts = []
    total_chars = 0

    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue

        if total_chars + len(text) > max_chars:
            break

        context_parts.append(text)
        total_chars += len(text)

    return "\n\n".join(context_parts)
