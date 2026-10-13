create extension if not exists vector;

create table if not exists chunks (
    id        bigserial primary key,
    source    text   not null,
    content   text   not null,
    embedding vector(384) not null          -- all-MiniLM-L6-v2 = 384 dims
);

-- No vector index on purpose: at demo-corpus size an exact scan is fast, and an
-- approximate index (IVFFlat) can miss true neighbours on a handful of rows.
-- For a larger corpus, add HNSW (or IVFFlat with lists sized to the row count):
--   create index on chunks using hnsw (embedding vector_cosine_ops);
drop index if exists chunks_embedding_idx;  -- older versions of this file created an IVFFlat index

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
