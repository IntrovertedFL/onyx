from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

from danswer.llm.llm_provider_options import fetch_models_for_provider

if TYPE_CHECKING:
    from danswer.db.models import LLMProvider as LLMProviderModel


class TestLLMRequest(BaseModel):
    # provider level
    exisitng_api_key: bool = False
    provider: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    custom_config: dict[str, str] | None = None

    # model level
    default_model_name: str
    fast_default_model_name: str | None = None


class LLMProviderDescriptor(BaseModel):
    """A descriptor for an LLM provider that can be safely viewed by
    non-admin users. Used when giving a list of available LLMs."""

    name: str
    provider: str
    model_names: list[str]
    default_model_name: str
    fast_default_model_name: str | None
    is_default_provider: bool | None
    display_model_names: list[str] | None

    @classmethod
    def from_model(
        cls, llm_provider_model: "LLMProviderModel"
    ) -> "LLMProviderDescriptor":
        return cls(
            name=llm_provider_model.name,
            provider=llm_provider_model.provider,
            default_model_name=llm_provider_model.default_model_name,
            fast_default_model_name=llm_provider_model.fast_default_model_name,
            is_default_provider=llm_provider_model.is_default_provider,
            model_names=(
                llm_provider_model.model_names
                or fetch_models_for_provider(llm_provider_model.provider)
                or [llm_provider_model.default_model_name]
            ),
            display_model_names=llm_provider_model.display_model_names,
        )


class LLMProvider(BaseModel):
    name: str
    provider: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    custom_config: dict[str, str] | None = None
    default_model_name: str
    fast_default_model_name: str | None = None
    is_public: bool = True
    groups: list[int] = Field(default_factory=list)
    display_model_names: list[str] | None = None


class LLMProviderUpsertRequest(LLMProvider):
    # should only be used for a "custom" provider
    # for default providers, the built-in model names are used
    model_names: list[str] | None = None


class LLMProviderUpdateRequest(LLMProvider):
    api_key_set: bool


class LLMProviderCreationRequest(LLMProvider):
    pass


class FullLLMProvider(LLMProvider):
    id: int
    is_default_provider: bool | None = None
    model_names: list[str]

    @classmethod
    def from_model(cls, llm_provider_model: "LLMProviderModel") -> "FullLLMProvider":
        return cls(
            api_key=llm_provider_model.api_key,
            id=llm_provider_model.id,
            name=llm_provider_model.name,
            provider=llm_provider_model.provider,
            api_base=llm_provider_model.api_base,
            api_version=llm_provider_model.api_version,
            custom_config=llm_provider_model.custom_config,
            default_model_name=llm_provider_model.default_model_name,
            fast_default_model_name=llm_provider_model.fast_default_model_name,
            is_default_provider=llm_provider_model.is_default_provider,
            display_model_names=llm_provider_model.display_model_names,
            model_names=(
                llm_provider_model.model_names
                or fetch_models_for_provider(llm_provider_model.provider)
                or [llm_provider_model.default_model_name]
            ),
            is_public=llm_provider_model.is_public,
            groups=[group.id for group in llm_provider_model.groups],
        )


class FullLLMProviderSnapshot(FullLLMProvider):
    api_key: None = None
    api_key_set: bool

    @classmethod
    def from_full_llm_provider(
        cls, settings: FullLLMProvider
    ) -> "FullLLMProviderSnapshot":
        data = settings.dict(exclude={"api_key"})
        data["api_key_set"] = bool(settings.api_key)
        return cls(**data)

    @classmethod
    def from_model(
        cls, llm_provider_model: "LLMProviderModel"
    ) -> "FullLLMProviderSnapshot":
        full_provider = FullLLMProvider.from_model(llm_provider_model)
        return cls.from_full_llm_provider(full_provider)
