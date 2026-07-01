create extension if not exists vector;

create table if not exists chunks (
    id        bigserial primary key,
    source    text   not null,
    content   text   not null,
    embedding vector(384) not null          -- all-MiniLM-L6-v2 = 384 dims
);

-- cosine-distance index; lists tuned for small corpora
create index if not exists chunks_embedding_idx
    on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 10);

-- top-k retrieval, returned as (source, content, similarity)
create or replace function match_chunks(query_embedding vector(384), k int)
returns table (source text, content text, similarity float)
language sql stable as $$
    select c.source,
           c.content,
           1 - (c.embedding <=> query_embedding) as similarity
    from   chunks c
    order  by c.embedding <=> query_embedding
    limit  k;
$$;
