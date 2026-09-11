from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.schemPost import UserResponse, UserCreate, PostResponse
from schemas.database import get_db
import schemas.model

router = APIRouter()

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Annotated[AsyncSession, Depends(get_db)]):
    existing_user = (await db.execute(
        select(schemas.model.User).where(schemas.model.User.username == user.username)
    )).scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        
    existing_email = (await db.execute(
        select(schemas.model.User).where(schemas.model.User.email == user.email)
    )).scalar_one_or_none()

    if existing_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    new_user = schemas.model.User(**user.model_dump())
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user_api(user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if user:
        return user
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

@router.get("/{user_id}/posts", response_model=list[PostResponse])
async def get_user_posts_api(user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    result = await db.execute(select(schemas.model.Post).where(schemas.model.Post.user_id == user_id))
    posts = result.scalars().all()
    return posts
