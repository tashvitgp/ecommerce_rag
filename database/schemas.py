from sqlalchemy import Column, String, Text, Numeric, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.connection import Base

class Product(Base):
    """
    Products dimensions table to house structural metadata. 
    """
    __tablename__ = "products"

    product_id = Column(String(255), primary_key=True, index=True)
    product_name = Column(Text)
    brand = Column(String(255), index=True)
    price = Column(Numeric(10, 2), index=True)
    category = Column(String(255), index=True)

    # Establish a relationship to easily fetch reviews for a product in Python
    reviews = relationship("Review", back_populates="product")

class Review(Base):
    """
    Reviews facts table for unstructured text and NLP aspect tags. 
    """
    __tablename__ = "reviews"

    review_id = Column(String(255), primary_key=True, index=True)
    product_id = Column(String(255), ForeignKey("products.product_id"))
    review_text = Column(Text)
    
    # Enriched fields for Agentic RAG
    cleaned_text = Column(Text)
    primary_aspect = Column(String(255), index=True)
    aspect_confidence = Column(Numeric(5, 4))
    
    rating = Column(Integer)
    timestamp = Column(DateTime)

    # Link back to the parent Product
    product = relationship("Product", back_populates="reviews")

class ConversationalCache(Base):
    """
    Engineering tracking table to store previously answered queries. 
    """
    __tablename__ = "conversational_cache"

    query_hash = Column(String(64), primary_key=True, index=True)
    original_query = Column(Text)
    cached_response = Column(Text)
    timestamp = Column(DateTime, server_default=func.now())