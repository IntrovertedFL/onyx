"use client";

import React, { useCallback } from "react";
import { Option } from "@/components/Dropdown";
import { generateRandomIconShape } from "@/lib/assistantIconUtils";
import { CCPairBasicInfo, DocumentSet, User } from "@/lib/types";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { IsPublicGroupSelector } from "@/components/IsPublicGroupSelector";
import { ArrayHelpers, FieldArray, Form, Formik, FormikProps } from "formik";

import {
  BooleanFormField,
  Label,
  SelectorFormField,
  TextFormField,
} from "@/components/admin/connectors/Field";

import { usePopup } from "@/components/admin/connectors/Popup";
import { getDisplayNameForModel, useCategories } from "@/lib/hooks";
import { DocumentSetSelectable } from "@/components/documentSet/DocumentSetSelectable";
import { addAssistantToList } from "@/lib/assistants/updateAssistantPreferences";
import { checkLLMSupportsImageInput, destructureValue } from "@/lib/llm/utils";
import { ToolSnapshot } from "@/lib/tools/interfaces";
import { checkUserIsNoAuthUser } from "@/lib/user";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { FiInfo, FiRefreshCcw } from "react-icons/fi";
import * as Yup from "yup";
import CollapsibleSection from "./CollapsibleSection";
import { SuccessfulPersonaUpdateRedirectType } from "./enums";
import { Persona, PersonaCategory, StarterMessage } from "./interfaces";
import {
  createPersonaCategory,
  createPersona,
  deletePersonaCategory,
  updatePersonaCategory,
  updatePersona,
} from "./lib";
import { Popover } from "@/components/popover/Popover";
import {
  CameraIcon,
  NewChatIcon,
  PlusIcon,
  SwapIcon,
  TrashIcon,
} from "@/components/icons/icons";
import { AdvancedOptionsToggle } from "@/components/AdvancedOptionsToggle";
import { buildImgUrl } from "@/app/chat/files/images/utils";
import { LlmList } from "@/components/llm/LLMList";
import { useAssistants } from "@/components/context/AssistantsContext";
import { debounce } from "lodash";
import { FullLLMProvider } from "../configuration/llm/interfaces";
import StarterMessagesList from "./StarterMessageList";
import { Input } from "@/components/ui/input";
import { CategoryCard } from "./CategoryCard";
import { Switch } from "@/components/ui/switch";
import { generateIdenticon } from "@/components/assistants/AssistantIcon";
import { BackButton } from "@/components/BackButton";
import { Checkbox } from "@/components/ui/checkbox";

function findSearchTool(tools: ToolSnapshot[]) {
  return tools.find((tool) => tool.in_code_tool_id === "SearchTool");
}

function findImageGenerationTool(tools: ToolSnapshot[]) {
  return tools.find((tool) => tool.in_code_tool_id === "ImageGenerationTool");
}

function findInternetSearchTool(tools: ToolSnapshot[]) {
  return tools.find((tool) => tool.in_code_tool_id === "InternetSearchTool");
}

function SubLabel({ children }: { children: string | JSX.Element }) {
  return (
    <div
      className="text-sm text-description font-description mb-2"
      style={{ color: "rgb(113, 114, 121)" }}
    >
      {children}
    </div>
  );
}

export function AssistantEditor({
  existingPersona,
  ccPairs,
  documentSets,
  user,
  defaultPublic,
  redirectType,
  llmProviders,
  tools,
  shouldAddAssistantToUserPreferences,
  admin,
}: {
  existingPersona?: Persona | null;
  ccPairs: CCPairBasicInfo[];
  documentSets: DocumentSet[];
  user: User | null;
  defaultPublic: boolean;
  redirectType: SuccessfulPersonaUpdateRedirectType;
  llmProviders: FullLLMProvider[];
  tools: ToolSnapshot[];
  shouldAddAssistantToUserPreferences?: boolean;
  admin?: boolean;
}) {
  const { refreshAssistants, isImageGenerationAvailable } = useAssistants();
  const router = useRouter();

  const { popup, setPopup } = usePopup();
  const { data: categories, refreshCategories } = useCategories();

  const colorOptions = [
    "#FF6FBF",
    "#6FB1FF",
    "#B76FFF",
    "#FFB56F",
    "#6FFF8D",
    "#FF6F6F",
    "#6FFFFF",
  ];

  const [showSearchTool, setShowSearchTool] = useState(false);

  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [hasEditedStarterMessage, setHasEditedStarterMessage] = useState(false);
  const [showPersonaCategory, setShowPersonaCategory] = useState(!admin);

  // state to persist across formik reformatting
  const [defautIconColor, _setDeafultIconColor] = useState(
    colorOptions[Math.floor(Math.random() * colorOptions.length)]
  );
  const [isRefreshing, setIsRefreshing] = useState(false);

  const [defaultIconShape, setDefaultIconShape] = useState<any>(null);

  useEffect(() => {
    if (defaultIconShape === null) {
      setDefaultIconShape(generateRandomIconShape().encodedGrid);
    }
  }, [defaultIconShape]);

  const [isIconDropdownOpen, setIsIconDropdownOpen] = useState(false);

  const [removePersonaImage, setRemovePersonaImage] = useState(false);

  const autoStarterMessageEnabled = useMemo(
    () => llmProviders.length > 0,
    [llmProviders.length]
  );
  const isUpdate = existingPersona !== undefined && existingPersona !== null;
  const existingPrompt = existingPersona?.prompts[0] ?? null;
  const defaultProvider = llmProviders.find(
    (llmProvider) => llmProvider.is_default_provider
  );
  const defaultModelName = defaultProvider?.default_model_name;
  const providerDisplayNameToProviderName = new Map<string, string>();
  llmProviders.forEach((llmProvider) => {
    providerDisplayNameToProviderName.set(
      llmProvider.name,
      llmProvider.provider
    );
  });

  const modelOptionsByProvider = new Map<string, Option<string>[]>();
  llmProviders.forEach((llmProvider) => {
    const providerOptions = llmProvider.model_names.map((modelName) => {
      return {
        name: getDisplayNameForModel(modelName),
        value: modelName,
      };
    });
    modelOptionsByProvider.set(llmProvider.name, providerOptions);
  });

  const personaCurrentToolIds =
    existingPersona?.tools.map((tool) => tool.id) || [];

  const searchTool = findSearchTool(tools);
  const imageGenerationTool = findImageGenerationTool(tools);
  const internetSearchTool = findInternetSearchTool(tools);

  const customTools = tools.filter(
    (tool) =>
      tool.in_code_tool_id !== searchTool?.in_code_tool_id &&
      tool.in_code_tool_id !== imageGenerationTool?.in_code_tool_id &&
      tool.in_code_tool_id !== internetSearchTool?.in_code_tool_id
  );

  const availableTools = [
    ...customTools,
    ...(searchTool ? [searchTool] : []),
    ...(imageGenerationTool ? [imageGenerationTool] : []),
    ...(internetSearchTool ? [internetSearchTool] : []),
  ];
  const enabledToolsMap: { [key: number]: boolean } = {};
  availableTools.forEach((tool) => {
    enabledToolsMap[tool.id] = personaCurrentToolIds.includes(tool.id);
  });

  const initialValues = {
    name: existingPersona?.name ?? "",
    description: existingPersona?.description ?? "",
    system_prompt: existingPrompt?.system_prompt ?? "",
    task_prompt: existingPrompt?.task_prompt ?? "",
    is_public: existingPersona?.is_public ?? defaultPublic,
    document_set_ids:
      existingPersona?.document_sets?.map((documentSet) => documentSet.id) ??
      ([] as number[]),
    num_chunks: existingPersona?.num_chunks ?? null,
    search_start_date: existingPersona?.search_start_date
      ? existingPersona?.search_start_date.toString().split("T")[0]
      : null,
    include_citations: existingPersona?.prompts[0]?.include_citations ?? true,
    llm_relevance_filter: existingPersona?.llm_relevance_filter ?? false,
    llm_model_provider_override:
      existingPersona?.llm_model_provider_override ?? null,
    llm_model_version_override:
      existingPersona?.llm_model_version_override ?? null,
    starter_messages: existingPersona?.starter_messages ?? [],
    enabled_tools_map: enabledToolsMap,
    icon_color: existingPersona?.icon_color ?? defautIconColor,
    icon_shape: existingPersona?.icon_shape ?? defaultIconShape,
    uploaded_image: null,
    category_id: existingPersona?.category_id ?? null,

    // EE Only
    groups: existingPersona?.groups ?? [],
  };

  interface AssistantPrompt {
    message: string;
    name: string;
  }

  const debouncedRefreshPrompts = debounce(
    async (formValues: any, setFieldValue: any) => {
      if (!autoStarterMessageEnabled) {
        return;
      }
      setIsRefreshing(true);
      console.log("Form values:", formValues);
      try {
        const response = await fetch("/api/persona/assistant-prompt-refresh", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name: formValues.name || "",
            description: formValues.description || "",
            document_set_ids: formValues.document_set_ids || [],
            instructions:
              formValues.system_prompt || formValues.task_prompt || "",
          }),
        });

        const data: AssistantPrompt[] = await response.json();
        if (response.ok) {
          setFieldValue("starter_messages", data);
        }
      } catch (error) {
        console.error("Failed to refresh prompts:", error);
      } finally {
        setIsRefreshing(false);
      }
    },
    1000
  );

  const [isRequestSuccessful, setIsRequestSuccessful] = useState(false);

  return (
    <div className="mx-auto max-w-4xl">
      <style>
        {`
          .assistant-editor input::placeholder,
          .assistant-editor textarea::placeholder {
            opacity: 0.5;
          }
        `}
      </style>
      <div className="absolute top-4 left-4">
        <BackButton />
      </div>
      {popup}
      <Formik
        enableReinitialize={true}
        initialValues={initialValues}
        validationSchema={Yup.object()
          .shape({
            name: Yup.string().required(
              "Must provide a name for the Assistant"
            ),
            description: Yup.string().required(
              "Must provide a description for the Assistant"
            ),
            system_prompt: Yup.string(),
            task_prompt: Yup.string(),
            is_public: Yup.boolean().required(),
            document_set_ids: Yup.array().of(Yup.number()),
            num_chunks: Yup.number().nullable(),
            include_citations: Yup.boolean().required(),
            llm_relevance_filter: Yup.boolean().required(),
            llm_model_version_override: Yup.string().nullable(),
            llm_model_provider_override: Yup.string().nullable(),
            starter_messages: Yup.array().of(
              Yup.object().shape({
                name: Yup.string().required("Must have a name"),
                message: Yup.string().required("Must have a message"),
              })
            ),
            search_start_date: Yup.date().nullable(),
            icon_color: Yup.string(),
            icon_shape: Yup.number(),
            uploaded_image: Yup.mixed().nullable(),
            category_id: Yup.number().nullable(),
            // EE Only
            groups: Yup.array().of(Yup.number()),
          })
          .test(
            "system-prompt-or-task-prompt",
            "Must provide either Instructions or Reminders (Advanced)",
            function (values) {
              const systemPromptSpecified =
                values.system_prompt && values.system_prompt.trim().length > 0;
              const taskPromptSpecified =
                values.task_prompt && values.task_prompt.trim().length > 0;

              if (systemPromptSpecified || taskPromptSpecified) {
                return true;
              }

              return this.createError({
                path: "system_prompt",
                message:
                  "Must provide either Instructions or Reminders (Advanced)",
              });
            }
          )}
        onSubmit={async (values, formikHelpers) => {
          if (
            values.llm_model_provider_override &&
            !values.llm_model_version_override
          ) {
            setPopup({
              type: "error",
              message:
                "Must select a model if a non-default LLM provider is chosen.",
            });
            return;
          }

          formikHelpers.setSubmitting(true);
          let enabledTools = Object.keys(values.enabled_tools_map)
            .map((toolId) => Number(toolId))
            .filter((toolId) => values.enabled_tools_map[toolId]);
          const searchToolEnabled = searchTool
            ? enabledTools.includes(searchTool.id)
            : false;
          const imageGenerationToolEnabled = imageGenerationTool
            ? enabledTools.includes(imageGenerationTool.id)
            : false;

          if (imageGenerationToolEnabled) {
            if (
              // model must support image input for image generation
              // to work
              !checkLLMSupportsImageInput(
                values.llm_model_version_override || defaultModelName || ""
              )
            ) {
              enabledTools = enabledTools.filter(
                (toolId) => toolId !== imageGenerationTool!.id
              );
            }
          }

          // if disable_retrieval is set, set num_chunks to 0
          // to tell the backend to not fetch any documents
          const numChunks = searchToolEnabled ? values.num_chunks || 10 : 0;

          // don't set groups if marked as public
          const groups = values.is_public ? [] : values.groups;

          let promptResponse;
          let personaResponse;
          if (isUpdate) {
            [promptResponse, personaResponse] = await updatePersona({
              id: existingPersona.id,
              existingPromptId: existingPrompt?.id,
              ...values,
              search_start_date: values.search_start_date
                ? new Date(values.search_start_date)
                : null,
              num_chunks: numChunks,
              users:
                user && !checkUserIsNoAuthUser(user.id) ? [user.id] : undefined,
              groups,
              tool_ids: enabledTools,
              remove_image: removePersonaImage,
            });
          } else {
            [promptResponse, personaResponse] = await createPersona({
              ...values,
              is_default_persona: admin!,
              num_chunks: numChunks,
              search_start_date: values.search_start_date
                ? new Date(values.search_start_date)
                : null,
              users:
                user && !checkUserIsNoAuthUser(user.id) ? [user.id] : undefined,
              groups,
              tool_ids: enabledTools,
            });
          }

          let error = null;
          if (!promptResponse.ok) {
            error = await promptResponse.text();
          }

          if (!personaResponse) {
            error = "Failed to create Assistant - no response received";
          } else if (!personaResponse.ok) {
            error = await personaResponse.text();
          }

          if (error || !personaResponse) {
            setPopup({
              type: "error",
              message: `Failed to create Assistant - ${error}`,
            });
            formikHelpers.setSubmitting(false);
          } else {
            const assistant = await personaResponse.json();
            const assistantId = assistant.id;
            if (
              shouldAddAssistantToUserPreferences &&
              user?.preferences?.chosen_assistants
            ) {
              const success = await addAssistantToList(assistantId);
              if (success) {
                setPopup({
                  message: `"${assistant.name}" has been added to your list.`,
                  type: "success",
                });
                await refreshAssistants();
              } else {
                setPopup({
                  message: `"${assistant.name}" could not be added to your list.`,
                  type: "error",
                });
              }
            }

            await refreshAssistants();
            router.push(
              redirectType === SuccessfulPersonaUpdateRedirectType.ADMIN
                ? `/admin/assistants?u=${Date.now()}`
                : `/chat?assistantId=${assistantId}`
            );
            setIsRequestSuccessful(true);
          }
        }}
      >
        {({
          isSubmitting,
          values,
          setFieldValue,
          errors,
          ...formikProps
        }: FormikProps<any>) => {
          function toggleToolInValues(toolId: number) {
            const updatedEnabledToolsMap = {
              ...values.enabled_tools_map,
              [toolId]: !values.enabled_tools_map[toolId],
            };
            setFieldValue("enabled_tools_map", updatedEnabledToolsMap);
          }

          // model must support image input for image generation
          // to work
          const currentLLMSupportsImageOutput = checkLLMSupportsImageInput(
            values.llm_model_version_override || defaultModelName || ""
          );

          return (
            <Form className="w-full text-text-950 assistant-editor">
              {/* Refresh starter messages when name or description changes */}
              <p className="text-base font-normal !text-2xl">
                {existingPersona ? (
                  <>
                    Edit assistant <b>{existingPersona.name}</b>
                  </>
                ) : (
                  "Create an Assistant"
                )}
              </p>
              <div className="max-w-4xl w-full">
                <Separator />
                <div className="flex gap-x-2 items-center">
                  <div className="block font-medium text-sm">
                    Assistant Icon
                  </div>
                </div>
                <SubLabel>
                  The icon that will visually represent your Assistant
                </SubLabel>
                <div className="flex gap-x-2 items-center">
                  <div
                    className="p-4 cursor-pointer  rounded-full flex  "
                    style={{
                      borderStyle: "dashed",
                      borderWidth: "1.5px",
                      borderSpacing: "4px",
                    }}
                  >
                    {values.uploaded_image ? (
                      <img
                        src={URL.createObjectURL(values.uploaded_image)}
                        alt="Uploaded assistant icon"
                        className="w-12 h-12 rounded-full object-cover"
                      />
                    ) : existingPersona?.uploaded_image_id &&
                      !removePersonaImage ? (
                      <img
                        src={buildImgUrl(existingPersona?.uploaded_image_id)}
                        alt="Uploaded assistant icon"
                        className="w-12 h-12 rounded-full object-cover"
                      />
                    ) : (
                      generateIdenticon((values.icon_shape || 0).toString(), 36)
                    )}
                  </div>

                  <div className="flex flex-col gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="text-xs flex justify-start gap-x-2"
                      onClick={() => {
                        const fileInput = document.createElement("input");
                        fileInput.type = "file";
                        fileInput.accept = "image/*";
                        fileInput.onchange = (e) => {
                          const file = (e.target as HTMLInputElement)
                            .files?.[0];
                          if (file) {
                            setFieldValue("uploaded_image", file);
                          }
                        };
                        fileInput.click();
                      }}
                    >
                      <CameraIcon size={14} />
                      Upload {values.uploaded_image && "New "}Image
                    </Button>

                    {values.uploaded_image && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="flex justify-start gap-x-2 text-xs"
                        onClick={() => {
                          setFieldValue("uploaded_image", null);
                          setRemovePersonaImage(false);
                        }}
                      >
                        <TrashIcon className="h-3 w-3" />
                        {removePersonaImage ? "Revert to Previous " : "Remove "}
                        Image
                      </Button>
                    )}

                    {!values.uploaded_image &&
                      (!existingPersona?.uploaded_image_id ||
                        removePersonaImage) && (
                        <Button
                          type="button"
                          variant="outline"
                          className="text-xs"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            const newShape = generateRandomIconShape();
                            const randomColor =
                              colorOptions[
                                Math.floor(Math.random() * colorOptions.length)
                              ];
                            setFieldValue("icon_shape", newShape.encodedGrid);
                            setFieldValue("icon_color", randomColor);
                          }}
                        >
                          <NewChatIcon size={14} />
                          Generate Icon
                        </Button>
                      )}

                    {existingPersona?.uploaded_image_id &&
                      removePersonaImage &&
                      !values.uploaded_image && (
                        <Button
                          type="button"
                          variant="outline"
                          className="text-xs"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setRemovePersonaImage(false);
                            setFieldValue("uploaded_image", null);
                          }}
                        >
                          <SwapIcon className="h-3 w-3" />
                          Revert to Previous Image
                        </Button>
                      )}

                    {existingPersona?.uploaded_image_id &&
                      !removePersonaImage &&
                      !values.uploaded_image && (
                        <Button
                          type="button"
                          variant="outline"
                          className="text-xs"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setRemovePersonaImage(true);
                          }}
                        >
                          <TrashIcon className="h-3 w-3" />
                          Remove Image
                        </Button>
                      )}
                  </div>
                </div>
              </div>

              <TextFormField
                maxWidth="max-w-lg"
                name="name"
                label="Name"
                placeholder="Email Assistant"
                aria-label="assistant-name-input"
                className="[&_input]:placeholder:text-text-muted/50"
              />

              <TextFormField
                maxWidth="max-w-lg"
                name="description"
                label="Description"
                placeholder="Use this Assistant to help draft professional emails"
                data-testid="assistant-description-input"
                className="[&_input]:placeholder:text-text-muted/50"
              />
              <Separator />

              <TextFormField
                maxWidth="max-w-4xl"
                name="system_prompt"
                label="Instructions"
                isTextArea={true}
                placeholder="You are a professional email writing assistant that always uses a polite enthusiastic tone, emphasizes action items, and leaves blanks for the human to fill in when you have unknowns"
                data-testid="assistant-instructions-input"
                className="[&_textarea]:placeholder:text-text-muted/50"
              />

              {llmProviders.length > 0 && (
                <>
                  <TextFormField
                    maxWidth="max-w-4xl"
                    name="task_prompt"
                    label="Reminders (Optional)"
                    isTextArea={true}
                    placeholder="Remember to reference all of the points mentioned in my message to you and focus on identifying action items that can move things forward"
                    onChange={(e) => {
                      setFieldValue("task_prompt", e.target.value);
                    }}
                    explanationText="Learn about prompting in our docs!"
                    explanationLink="https://docs.onyx.app/guides/assistants"
                    className="[&_textarea]:placeholder:text-text-muted/50"
                  />
                </>
              )}
              <div className="w-full max-w-4xl">
                <div className="flex flex-col">
                  {searchTool && (
                    <>
                      <Separator />
                      <div className="flex gap-x-2 py-2 flex justify-start">
                        <div>
                          <div
                            className="flex items-start gap-x-2
                          "
                          >
                            <p className="block font-medium text-sm">
                              Knowledge
                            </p>
                            <div className="flex items-center">
                              <TooltipProvider delayDuration={0}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <div
                                      className={`${
                                        ccPairs.length === 0
                                          ? "opacity-70 cursor-not-allowed"
                                          : ""
                                      }`}
                                    >
                                      <Switch
                                        size="sm"
                                        onCheckedChange={(checked) => {
                                          setShowSearchTool(checked);
                                          setFieldValue("num_chunks", null);
                                          toggleToolInValues(searchTool.id);
                                        }}
                                        name={`enabled_tools_map.${searchTool.id}`}
                                        disabled={ccPairs.length === 0}
                                      />
                                    </div>
                                  </TooltipTrigger>

                                  {ccPairs.length === 0 && (
                                    <TooltipContent side="top" align="center">
                                      <p className="bg-background-900 max-w-[200px] text-sm rounded-lg p-1.5 text-white">
                                        To use the Knowledge Action, you need to
                                        have at least one Connector-Credential
                                        pair configured.
                                      </p>
                                    </TooltipContent>
                                  )}
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                          </div>
                          <p
                            className="text-sm text-subtle"
                            style={{ color: "rgb(113, 114, 121)" }}
                          >
                            Enable search capabilities for this assistant
                          </p>
                        </div>
                      </div>
                    </>
                  )}
                  {ccPairs.length > 0 &&
                    searchTool &&
                    showSearchTool &&
                    !(user?.role != "admin" && documentSets.length === 0) && (
                      <div className="mt-2">
                        {ccPairs.length > 0 && (
                          <>
                            <Label small>Document Sets</Label>
                            <div>
                              <SubLabel>
                                <>
                                  Select which{" "}
                                  {!user || user.role === "admin" ? (
                                    <Link
                                      href="/admin/documents/sets"
                                      className="font-semibold hover:underline text-text"
                                      target="_blank"
                                    >
                                      Document Sets
                                    </Link>
                                  ) : (
                                    "Document Sets"
                                  )}{" "}
                                  this Assistant should use to inform its
                                  responses. If none are specified, the
                                  Assistant will reference all available
                                  documents.
                                </>
                              </SubLabel>
                            </div>

                            {documentSets.length > 0 ? (
                              <FieldArray
                                name="document_set_ids"
                                render={(arrayHelpers: ArrayHelpers) => (
                                  <div>
                                    <div className="mb-3 mt-2 flex gap-2 flex-wrap text-sm">
                                      {documentSets.map((documentSet) => (
                                        <DocumentSetSelectable
                                          key={documentSet.id}
                                          documentSet={documentSet}
                                          isSelected={values.document_set_ids.includes(
                                            documentSet.id
                                          )}
                                          onSelect={() => {
                                            const index =
                                              values.document_set_ids.indexOf(
                                                documentSet.id
                                              );
                                            if (index !== -1) {
                                              arrayHelpers.remove(index);
                                            } else {
                                              arrayHelpers.push(documentSet.id);
                                            }
                                          }}
                                        />
                                      ))}
                                    </div>
                                  </div>
                                )}
                              />
                            ) : (
                              <p className="text-sm flex gap-x-2">
                                <button
                                  type="button"
                                  onClick={() =>
                                    router.push("/admin/documents/sets/new")
                                  }
                                  className="py-1 px-2 rounded-md bg-black"
                                >
                                  <PlusIcon
                                    className="bg-black text-white"
                                    size={12}
                                  />
                                </button>
                                Create a document set to get started.
                              </p>
                            )}

                            <div className="mt-4 flex flex-col gap-y-4">
                              <TextFormField
                                small={true}
                                name="num_chunks"
                                label="Number of Context Documents"
                                placeholder="Defaults to 10"
                                onChange={(e) => {
                                  const value = e.target.value;
                                  if (value === "" || /^[0-9]+$/.test(value)) {
                                    setFieldValue("num_chunks", value);
                                  }
                                }}
                              />

                              <TextFormField
                                width="max-w-xl"
                                type="date"
                                small
                                subtext="Documents prior to this date will not be ignored."
                                optional
                                label="Knowledge Cutoff Date"
                                value={values.search_start_date}
                                name="search_start_date"
                              />

                              <BooleanFormField
                                small
                                removeIndent
                                alignTop
                                name="llm_relevance_filter"
                                label="AI Relevance Filter"
                                subtext="If enabled, the LLM will filter out documents that are not useful for answering the user query prior to generating a response. This typically improves the quality of the response but incurs slightly higher cost."
                              />

                              <BooleanFormField
                                small
                                removeIndent
                                alignTop
                                name="include_citations"
                                label="Citations"
                                subtext="Response will include citations ([1], [2], etc.) for documents referenced by the LLM. In general, we recommend to leave this enabled in order to increase trust in the LLM answer."
                              />
                            </div>
                          </>
                        )}
                      </div>
                    )}

                  <Separator />
                  <div className="py-2">
                    <p className="block font-medium text-sm mb-2">Actions</p>

                    {imageGenerationTool && (
                      <>
                        <div className="flex items-center content-start mb-2">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger>
                                <Checkbox
                                  id={`enabled_tools_map.${imageGenerationTool.id}`}
                                  checked={
                                    values.enabled_tools_map[
                                      imageGenerationTool.id
                                    ]
                                  }
                                  onCheckedChange={() => {
                                    toggleToolInValues(imageGenerationTool.id);
                                  }}
                                  disabled={
                                    !currentLLMSupportsImageOutput ||
                                    !isImageGenerationAvailable
                                  }
                                />
                              </TooltipTrigger>
                              {(!currentLLMSupportsImageOutput ||
                                !isImageGenerationAvailable) && (
                                <TooltipContent side="top" align="center">
                                  <p className="bg-background-900 max-w-[200px] mb-1 text-sm rounded-lg p-1.5 text-white">
                                    {!currentLLMSupportsImageOutput
                                      ? "To use Image Generation, select GPT-4 or another image compatible model as the default model for this Assistant."
                                      : "Image Generation requires an OpenAI or Azure Dalle configuration."}
                                  </p>
                                </TooltipContent>
                              )}
                            </Tooltip>
                          </TooltipProvider>
                          <div className="ml-2">
                            <span className="text-sm">
                              {imageGenerationTool.display_name}
                            </span>
                          </div>
                        </div>
                      </>
                    )}

                    {internetSearchTool && (
                      <>
                        <div className="flex items-center content-start mb-2">
                          <Checkbox
                            id={`enabled_tools_map.${internetSearchTool.id}`}
                            checked={
                              values.enabled_tools_map[internetSearchTool.id]
                            }
                            onCheckedChange={() => {
                              toggleToolInValues(internetSearchTool.id);
                            }}
                          />
                          <div className="ml-2">
                            <span className="text-sm">
                              {internetSearchTool.display_name}
                            </span>
                          </div>
                        </div>
                      </>
                    )}

                    {customTools.length > 0 &&
                      customTools.map((tool) => (
                        <React.Fragment key={tool.id}>
                          <div className="flex items-center content-start mb-2">
                            <Checkbox
                              id={`enabled_tools_map.${tool.id}`}
                              checked={values.enabled_tools_map[tool.id]}
                              onCheckedChange={() => {
                                toggleToolInValues(tool.id);
                              }}
                            />
                            <div className="ml-2">
                              <span className="text-sm">
                                {tool.display_name}
                              </span>
                            </div>
                          </div>
                        </React.Fragment>
                      ))}
                  </div>
                </div>
              </div>
              <Separator className="max-w-4xl mt-0" />
              <div className="-mt-2">
                <div className="flex gap-x-2 items-center">
                  <div className="block  font-medium text-sm">
                    Default AI Model{" "}
                  </div>
                </div>

                {admin ? (
                  <div className="mb-2 flex items-starts">
                    <div className="w-96">
                      <SelectorFormField
                        defaultValue={`User default`}
                        name="llm_model_provider_override"
                        options={llmProviders.map((llmProvider) => ({
                          name: llmProvider.name,
                          value: llmProvider.name,
                          icon: llmProvider.icon,
                        }))}
                        includeDefault={true}
                        onSelect={(selected) => {
                          if (selected !== values.llm_model_provider_override) {
                            setFieldValue("llm_model_version_override", null);
                          }
                          setFieldValue(
                            "llm_model_provider_override",
                            selected
                          );
                        }}
                      />
                    </div>

                    {values.llm_model_provider_override && (
                      <div className="w-96 ml-4">
                        <SelectorFormField
                          name="llm_model_version_override"
                          options={
                            modelOptionsByProvider.get(
                              values.llm_model_provider_override
                            ) || []
                          }
                          maxHeight="max-h-72"
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <LlmList
                    scrollable
                    userDefault={
                      user?.preferences?.default_model!
                        ? destructureValue(user?.preferences?.default_model!)
                            .modelName
                        : null
                    }
                    llmProviders={llmProviders}
                    currentLlm={values.llm_model_version_override}
                    onSelect={(value: string | null) => {
                      if (value !== null) {
                        const { modelName, provider, name } =
                          destructureValue(value);
                        setFieldValue("llm_model_version_override", modelName);
                        setFieldValue("llm_model_provider_override", name);
                      } else {
                        setFieldValue("llm_model_version_override", null);
                        setFieldValue("llm_model_provider_override", null);
                      }
                    }}
                  />
                )}
              </div>
              <Separator className="max-w-4xl" />
              <div className="w-full flex flex-col">
                <div className="flex gap-x-2 items-center">
                  <div className="block font-medium text-sm">
                    Starter Messages
                  </div>
                </div>

                <SubLabel>
                  Pre-configured messages that help users understand what this
                  assistant can do and how to interact with it effectively.
                </SubLabel>

                <div className="w-full">
                  <FieldArray
                    name="starter_messages"
                    render={(arrayHelpers: ArrayHelpers) => (
                      <StarterMessagesList
                        debouncedRefreshPrompts={() =>
                          debouncedRefreshPrompts(values, setFieldValue)
                        }
                        autoStarterMessageEnabled={autoStarterMessageEnabled}
                        errors={errors}
                        isRefreshing={isRefreshing}
                        values={values.starter_messages}
                        arrayHelpers={arrayHelpers}
                        touchStarterMessages={() => {
                          setHasEditedStarterMessage(true);
                        }}
                        setFieldValue={setFieldValue}
                      />
                    )}
                  />
                </div>
              </div>

              <>
                {categories && categories.length > 0 && (
                  <div className="my-2">
                    <div className="flex gap-x-2 items-center">
                      <div className="block font-medium text-base">
                        Category
                      </div>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <FiInfo size={12} />
                          </TooltipTrigger>
                          <TooltipContent side="top" align="center">
                            Group similar assistants together by category
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <SelectorFormField
                      includeReset
                      name="category_id"
                      options={categories.map((category) => ({
                        name: category.name,
                        value: category.id,
                      }))}
                    />
                  </div>
                )}

                {admin && (
                  <>
                    <div className="my-2">
                      <div className="flex gap-x-2 items-center mb-2">
                        <div className="block font-medium text-base">
                          Create New Category
                        </div>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <FiInfo size={12} />
                            </TooltipTrigger>
                            <TooltipContent side="top" align="center">
                              Create a new category to group similar assistants
                              together
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>

                      <div className="grid grid-cols-[1fr,3fr,auto] gap-4">
                        <TextFormField
                          fontSize="sm"
                          name="newCategoryName"
                          label="Category Name"
                          placeholder="Development"
                        />
                        <TextFormField
                          fontSize="sm"
                          name="newCategoryDescription"
                          label="Category Description"
                          placeholder="Assistants for software development"
                        />

                        <div className="flex items-end">
                          <Button
                            type="button"
                            onClick={async () => {
                              const name = values.newCategoryName;
                              const description = values.newCategoryDescription;
                              if (!name || !description) return;

                              try {
                                const response = await createPersonaCategory(
                                  name,
                                  description
                                );
                                if (response.ok) {
                                  setPopup({
                                    message: `Category "${name}" created successfully`,
                                    type: "success",
                                  });
                                } else {
                                  throw new Error(await response.text());
                                }
                              } catch (error) {
                                setPopup({
                                  message: `Failed to create category - ${error}`,
                                  type: "error",
                                });
                              }

                              await refreshCategories();

                              setFieldValue("newCategoryName", "");
                              setFieldValue("newCategoryDescription", "");
                            }}
                          >
                            Create
                          </Button>
                        </div>
                      </div>
                    </div>

                    {categories && categories.length > 0 && (
                      <div className="my-2 w-full">
                        <div className="flex gap-x-2 items-center mb-2">
                          <div className="block font-medium text-base">
                            Manage categories
                          </div>
                          <TooltipProvider delayDuration={0}>
                            <Tooltip>
                              <TooltipTrigger>
                                <FiInfo size={12} />
                              </TooltipTrigger>
                              <TooltipContent side="top" align="center">
                                Manage existing categories or create new ones to
                                group similar assistants
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>
                        <div className="gap-4 w-full flex-wrap flex">
                          {categories &&
                            categories.map((category: PersonaCategory) => (
                              <CategoryCard
                                setPopup={setPopup}
                                key={category.id}
                                category={category}
                                onUpdate={async (id, name, description) => {
                                  const response = await updatePersonaCategory(
                                    id,
                                    name,
                                    description
                                  );
                                  if (response?.ok) {
                                    setPopup({
                                      message: `Category "${name}" updated successfully`,
                                      type: "success",
                                    });
                                  } else {
                                    setPopup({
                                      message: `Failed to update category - ${await response.text()}`,
                                      type: "error",
                                    });
                                  }
                                }}
                                onDelete={async (id) => {
                                  const response = await deletePersonaCategory(
                                    id
                                  );
                                  if (response?.ok) {
                                    setPopup({
                                      message: `Category deleted successfully`,
                                      type: "success",
                                    });
                                  } else {
                                    setPopup({
                                      message: `Failed to delete category - ${await response.text()}`,
                                      type: "error",
                                    });
                                  }
                                }}
                                refreshCategories={refreshCategories}
                              />
                            ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </>

              <div className="max-w-4xl w-full">
                <IsPublicGroupSelector
                  formikProps={{
                    values,
                    isSubmitting,
                    setFieldValue,
                    errors,
                    ...formikProps,
                  }}
                  objectName="assistant"
                  enforceGroupSelection={false}
                />

                <div className="mt-12 flex">
                  <Button
                    variant="submit"
                    type="submit"
                    disabled={isSubmitting || isRequestSuccessful}
                  >
                    {isUpdate ? "Update!" : "Create!"}
                  </Button>
                </div>
              </div>
            </Form>
          );
        }}
      </Formik>
    </div>
  );
}
