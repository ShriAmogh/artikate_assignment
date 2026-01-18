import os
from accounts.models import UserIndex

def get_or_create_user_index(user):
    collection_name = f"user_{user.username}_rag"
    chroma_dir = os.path.join("vector_store", collection_name)

    os.makedirs(chroma_dir, exist_ok=True)

    user_index, created = UserIndex.objects.get_or_create(
        user=user,
        defaults={
            "collection_name": collection_name,
            #"chroma_dir" : chroma_dir
        }
    )

    return user_index
