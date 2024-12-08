"use client";

import {
  FiEdit2,
  FiChevronRight,
  FiChevronLeft,
  FiTool,
  FiGlobe,
} from "react-icons/fi";
import { FeedbackType } from "../types";
import React, {
  memo,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import {
  DanswerDocument,
  FilteredDanswerDocument,
  LoadedDanswerDocument,
} from "@/lib/search/interfaces";
import { SearchSummary } from "./SearchSummary";

import { SkippedSearch } from "./SkippedSearch";
import remarkGfm from "remark-gfm";
import { CopyButton } from "@/components/CopyButton";
import { ChatFileType, FileDescriptor, ToolCallMetadata } from "../interfaces";
import {
  IMAGE_GENERATION_TOOL_NAME,
  SEARCH_TOOL_NAME,
  INTERNET_SEARCH_TOOL_NAME,
} from "../tools/constants";
import { ToolRunDisplay } from "../tools/ToolRunningAnimation";
import { Hoverable, HoverableIcon } from "@/components/Hoverable";
import { DocumentPreview } from "../files/documents/DocumentPreview";
import { InMessageImage } from "../files/images/InMessageImage";
import { CodeBlock } from "./CodeBlock";
import rehypePrism from "rehype-prism-plus";

import "prismjs/themes/prism-tomorrow.css";
import "./custom-code-styles.css";
import { Persona } from "@/app/admin/assistants/interfaces";
import { AssistantIcon } from "@/components/assistants/AssistantIcon";

import { LikeFeedback, DislikeFeedback } from "@/components/icons/icons";
import {
  CustomTooltip,
  TooltipGroup,
} from "@/components/tooltip/CustomTooltip";
import { ValidSources } from "@/lib/types";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useMouseTracking } from "./hooks";
import { SettingsContext } from "@/components/settings/SettingsProvider";
import GeneratingImageDisplay from "../tools/GeneratingImageDisplay";
import RegenerateOption from "../RegenerateOption";
import { LlmOverride } from "@/lib/hooks";
import { ContinueGenerating } from "./ContinueMessage";
import { MemoizedAnchor, MemoizedParagraph } from "./MemoizedTextComponents";
import { extractCodeText } from "./codeUtils";
import ToolResult from "../../../components/tools/ToolResult";
import CsvContent from "../../../components/tools/CSVContent";
import SourceCard, {
  SeeMoreBlock,
} from "@/components/chat_search/sources/SourceCard";

const TOOLS_WITH_CUSTOM_HANDLING = [
  SEARCH_TOOL_NAME,
  INTERNET_SEARCH_TOOL_NAME,
  IMAGE_GENERATION_TOOL_NAME,
];

function FileDisplay({
  files,
  alignBubble,
}: {
  files: FileDescriptor[];
  alignBubble?: boolean;
}) {
  const [close, setClose] = useState(true);
  const imageFiles = files.filter((file) => file.type === ChatFileType.IMAGE);
  const nonImgFiles = files.filter(
    (file) => file.type !== ChatFileType.IMAGE && file.type !== ChatFileType.CSV
  );

  const csvImgFiles = files.filter((file) => file.type == ChatFileType.CSV);

  return (
    <>
      {nonImgFiles && nonImgFiles.length > 0 && (
        <div
          id="danswer-file"
          className={` ${alignBubble && "ml-auto"} mt-2 auto mb-4`}
        >
          <div className="flex flex-col gap-2">
            {nonImgFiles.map((file) => {
              return (
                <div key={file.id} className="w-fit">
                  <DocumentPreview
                    fileName={file.name || file.id}
                    maxWidth="max-w-64"
                    alignBubble={alignBubble}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {imageFiles && imageFiles.length > 0 && (
        <div
          id="danswer-image"
          className={` ${alignBubble && "ml-auto"} mt-2 auto mb-4`}
        >
          <div className="flex flex-col gap-2">
            {imageFiles.map((file) => {
              return <InMessageImage key={file.id} fileId={file.id} />;
            })}
          </div>
        </div>
      )}

      {csvImgFiles && csvImgFiles.length > 0 && (
        <div className={` ${alignBubble && "ml-auto"} mt-2 auto mb-4`}>
          <div className="flex flex-col gap-2">
            {csvImgFiles.map((file) => {
              return (
                <div key={file.id} className="w-fit">
                  {close ? (
                    <>
                      <ToolResult
                        csvFileDescriptor={file}
                        close={() => setClose(false)}
                        contentComponent={CsvContent}
                      />
                    </>
                  ) : (
                    <DocumentPreview
                      open={() => setClose(true)}
                      fileName={file.name || file.id}
                      maxWidth="max-w-64"
                      alignBubble={alignBubble}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}

export const AIMessage = ({ humanText }: { humanText: string }) => {
  // This message layout is directly taken from the Figma snippet provided.
  // We'll render a static "Agentic Search" example to mimic the design.
  // In a real scenario, you would dynamically insert content.

  return (
    <div className="py-5 px-4 sm:px-6 md:px-8 lg:px-10 w-full max-w-full">
      <div className="mx-auto w-full max-w-7xl">
        <div className="flex flex-col md:flex-row">
          {/* Assistant icon */}
          <div className="flex-shrink-0 mb-4 md:mb-0 md:mr-6">
            <div className="w-10 h-10 p-1 flex justify-center items-center">
              <img
                className="w-full h-full object-cover rounded-full"
                src="https://via.placeholder.com/40x40"
                alt="assistant"
              />
            </div>
          </div>

          <div className="flex-grow">
            <div className="space-y-8 text-black font-['KH Teka TRIAL']">
              {/* The Question */}
              <h2 className="text-2xl sm:text-3xl md:text-4xl font-light leading-tight">
                {humanText}
              </h2>

              {/* Main answer container */}
              <div className="p-4 rounded border border-[#e6e3dd] bg-white">
                <div className="flex flex-col space-y-6">
                  <h3 className="text-xl font-semibold">Agentic Search</h3>
                  <hr className="border-[#e6e3dd]" />

                  {/* Getting context section */}
                  <div className="space-y-4">
                    <h4 className="font-medium">
                      Getting context on typical sales processes
                    </h4>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2 text-sm">
                        <span className="text-[#4a4a4a]">Searching</span>
                        <div className="flex items-center space-x-1">
                          <div className="w-6 h-6 p-1 rounded-xl border border-black flex justify-center items-center">
                            <div className="w-4 h-4" />
                          </div>
                          <span>Web Search</span>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {[
                          "Sales training materials",
                          "Sales playbook",
                          "Customer Journey",
                        ].map((item, index) => (
                          <span
                            key={index}
                            className={`px-2 py-1 text-xs rounded-2xl ${
                              index === 2 ? "bg-[#e6e3dd]" : "bg-[#f1eee8]"
                            }`}
                          >
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <span className="text-sm text-[#4a4a4a]">Reading</span>
                      <div className="flex flex-wrap gap-2">
                        {[
                          "2024 sales process onboarding guide",
                          "2024 onyx sales playbook",
                          "Ramp customer journey record",
                        ].map((item, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 text-xs bg-[#f1eee8] rounded-2xl"
                          >
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Ramp difference section */}
                  <div className="space-y-4">
                    <h4 className="font-medium">
                      What was done differently at Ramp
                    </h4>
                    <div className="space-y-2">
                      <span className="text-sm text-[#4a4a4a]">Searching</span>
                      <div className="flex flex-wrap gap-2">
                        {[
                          "Ramp sales calls",
                          "Ramp deal sales email exchanges",
                          "Ramp win/loss analysis",
                        ].map((item, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 text-xs bg-[#f1eee8] rounded-2xl"
                          >
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <span className="text-sm text-[#4a4a4a]">Reading</span>
                      <div className="flex flex-wrap gap-2">
                        {[
                          "Sales call with Ramp 2024/5/15",
                          "Sales call with Ramp 2024/4/16",
                          "Sales call with Ramp 2024/3/12",
                          "Ramp:Onyx Regarding the deal terms",
                          "Ramp:Onyx deal progress tracker",
                        ].map((item, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 text-xs bg-[#f1eee8] rounded-2xl"
                          >
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sources */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Sources</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {[
                    {
                      text: "a new in-app support experience for Ramp (recently rolled out to 50% of",
                      source: "Business Overview meeting",
                    },
                    {
                      text: "Danswer Updates!!! Shiny New Things for Ramp! General Improvements APIs for Ramp use",
                      source: "Ramp (April)",
                    },
                    {
                      text: "Existing solutions are so expensive that this isn't feasible. * Today's solutions don't meet those",
                      source: "Sales Pitch Deck Thoughts",
                    },
                    {
                      text: "Show All",
                      source: "",
                    },
                  ].map((item, index) => (
                    <div
                      key={index}
                      className="p-3 bg-[#f1eee8] rounded-lg space-y-2"
                    >
                      <p className="text-sm">{item.text}</p>
                      {item.source && (
                        <div className="flex items-center text-xs text-[#4a4a4a]">
                          <div className="w-4 h-4 mr-1" />
                          {item.source}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Conclusion */}
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 p-1 flex justify-center items-center">
                    <img
                      className="w-full h-full object-cover rounded-full"
                      src="https://via.placeholder.com/32x32"
                      alt="Onyx logo"
                    />
                  </div>
                  <h3 className="text-xl font-semibold">Onyx</h3>
                </div>
                <p className="text-base">
                  To win the deal at Ramp, the team made the product really easy
                  to access. After a successful demo...
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// export const AIMessage = ({
//   regenerate,
//   overriddenModel,
//   selectedMessageForDocDisplay,
//   continueGenerating,
//   shared,
//   isActive,
//   toggleDocumentSelection,
//   alternativeAssistant,
//   docs,
//   messageId,
//   documentSelectionToggled,
//   content,
//   files,
//   selectedDocuments,
//   query,
//   citedDocuments,
//   toolCall,
//   isComplete,
//   hasDocs,
//   handleFeedback,
//   handleShowRetrieved,
//   handleSearchQueryEdit,
//   handleForceSearch,
//   retrievalDisabled,
//   currentPersona,
//   otherMessagesCanSwitchTo,
//   onMessageSelection,
//   setPresentingDocument,
//   index,
// }: {
//   index?: number;
//   selectedMessageForDocDisplay?: number | null;
//   shared?: boolean;
//   isActive?: boolean;
//   continueGenerating?: () => void;
//   otherMessagesCanSwitchTo?: number[];
//   onMessageSelection?: (messageId: number) => void;
//   selectedDocuments?: DanswerDocument[] | null;
//   toggleDocumentSelection?: () => void;
//   docs?: DanswerDocument[] | null;
//   alternativeAssistant?: Persona | null;
//   currentPersona: Persona;
//   messageId: number | null;
//   content: string | JSX.Element;
//   documentSelectionToggled?: boolean;
//   files?: FileDescriptor[];
//   query?: string;
//   citedDocuments?: [string, DanswerDocument][] | null;
//   toolCall?: ToolCallMetadata | null;
//   isComplete?: boolean;
//   hasDocs?: boolean;
//   handleFeedback?: (feedbackType: FeedbackType) => void;
//   handleShowRetrieved?: (messageNumber: number | null) => void;
//   handleSearchQueryEdit?: (query: string) => void;
//   handleForceSearch?: () => void;
//   retrievalDisabled?: boolean;
//   overriddenModel?: string;
//   regenerate?: (modelOverRide: LlmOverride) => Promise<void>;
//   setPresentingDocument?: (document: DanswerDocument) => void;
// }) => {
//   const toolCallGenerating = toolCall && !toolCall.tool_result;
//   const processContent = (content: string | JSX.Element) => {
//     if (typeof content !== "string") {
//       return content;
//     }

//     const codeBlockRegex = /```(\w*)\n[\s\S]*?```|```[\s\S]*?$/g;
//     const matches = content.match(codeBlockRegex);

//     if (matches) {
//       content = matches.reduce((acc, match) => {
//         if (!match.match(/```\w+/)) {
//           return acc.replace(match, match.replace("```", "```plaintext"));
//         }
//         return acc;
//       }, content);

//       const lastMatch = matches[matches.length - 1];
//       if (!lastMatch.endsWith("```")) {
//         return content;
//       }
//     }

//     return content + (!isComplete && !toolCallGenerating ? " [*]() " : "");
//   };
//   const finalContent = processContent(content as string);

//   const [isRegenerateHovered, setIsRegenerateHovered] = useState(false);
//   const [isRegenerateDropdownVisible, setIsRegenerateDropdownVisible] =
//     useState(false);
//   const { isHovering, trackedElementRef, hoverElementRef } = useMouseTracking();

//   const settings = useContext(SettingsContext);
//   // this is needed to give Prism a chance to load

//   const selectedDocumentIds =
//     selectedDocuments?.map((document) => document.document_id) || [];
//   const citedDocumentIds: string[] = [];

//   citedDocuments?.forEach((doc) => {
//     citedDocumentIds.push(doc[1].document_id);
//   });

//   if (!isComplete) {
//     const trimIncompleteCodeSection = (
//       content: string | JSX.Element
//     ): string | JSX.Element => {
//       if (typeof content === "string") {
//         const pattern = /```[a-zA-Z]+[^\s]*$/;
//         const match = content.match(pattern);
//         if (match && match.index && match.index > 3) {
//           const newContent = content.slice(0, match.index - 3);
//           return newContent;
//         }
//         return content;
//       }
//       return content;
//     };
//     content = trimIncompleteCodeSection(content);
//   }

//   let filteredDocs: FilteredDanswerDocument[] = [];

//   if (docs) {
//     filteredDocs = docs
//       .filter(
//         (doc, index, self) =>
//           doc.document_id &&
//           doc.document_id !== "" &&
//           index === self.findIndex((d) => d.document_id === doc.document_id)
//       )
//       .filter((doc) => {
//         return citedDocumentIds.includes(doc.document_id);
//       })
//       .map((doc: DanswerDocument, ind: number) => {
//         return {
//           ...doc,
//           included: selectedDocumentIds.includes(doc.document_id),
//         };
//       });
//   }

//   const paragraphCallback = useCallback(
//     (props: any) => <MemoizedParagraph>{props.children}</MemoizedParagraph>,
//     []
//   );

//   const anchorCallback = useCallback(
//     (props: any) => (
//       <MemoizedAnchor
//         updatePresentingDocument={setPresentingDocument}
//         docs={docs}
//       >
//         {props.children}
//       </MemoizedAnchor>
//     ),
//     [docs]
//   );

//   const currentMessageInd = messageId
//     ? otherMessagesCanSwitchTo?.indexOf(messageId)
//     : undefined;

//   const uniqueSources: ValidSources[] = Array.from(
//     new Set((docs || []).map((doc) => doc.source_type))
//   ).slice(0, 3);

//   const markdownComponents = useMemo(
//     () => ({
//       a: anchorCallback,
//       p: paragraphCallback,
//       code: ({ node, className, children }: any) => {
//         const codeText = extractCodeText(
//           node,
//           finalContent as string,
//           children
//         );

//         return (
//           <CodeBlock className={className} codeText={codeText}>
//             {children}
//           </CodeBlock>
//         );
//       },
//     }),
//     [anchorCallback, paragraphCallback, finalContent]
//   );

//   const renderedMarkdown = useMemo(() => {
//     return (
//       <ReactMarkdown
//         className="prose max-w-full text-base"
//         components={markdownComponents}
//         remarkPlugins={[remarkGfm]}
//         rehypePlugins={[[rehypePrism, { ignoreMissing: true }]]}
//       >
//         {finalContent as string}
//       </ReactMarkdown>
//     );
//   }, [finalContent, markdownComponents]);

//   const includeMessageSwitcher =
//     currentMessageInd !== undefined &&
//     onMessageSelection &&
//     otherMessagesCanSwitchTo &&
//     otherMessagesCanSwitchTo.length > 1;
//   return (
//     <div
//       id="danswer-ai-message"
//       ref={trackedElementRef}
//       className={`py-5 ml-4 px-5 relative flex `}
//     >
//       <div
//         className={`mx-auto ${
//           shared ? "w-full" : "w-[90%]"
//         }  max-w-message-max`}
//       >
//         <div className={`desktop:mr-12 ${!shared && "mobile:ml-0 md:ml-8"}`}>
//           <div className="flex">
//             <AssistantIcon
//               size="small"
//               assistant={alternativeAssistant || currentPersona}
//             />

//             <div className="w-full">
//               <div className="max-w-message-max break-words">
//                 <div className="w-full ml-4">
//                   <div className="max-w-message-max break-words">
//                     {!toolCall || toolCall.tool_name === SEARCH_TOOL_NAME ? (
//                       <>
//                         {query !== undefined &&
//                           handleShowRetrieved !== undefined &&
//                           !retrievalDisabled && (
//                             <div className="mb-1">
//                               <SearchSummary
//                                 index={index || 0}
//                                 query={query}
//                                 finished={toolCall?.tool_result != undefined}
//                                 hasDocs={hasDocs || false}
//                                 messageId={messageId}
//                                 handleShowRetrieved={handleShowRetrieved}
//                                 handleSearchQueryEdit={handleSearchQueryEdit}
//                               />
//                             </div>
//                           )}
//                         {handleForceSearch &&
//                           content &&
//                           query === undefined &&
//                           !hasDocs &&
//                           !retrievalDisabled && (
//                             <div className="mb-1">
//                               <SkippedSearch
//                                 handleForceSearch={handleForceSearch}
//                               />
//                             </div>
//                           )}
//                       </>
//                     ) : null}

//                     {toolCall &&
//                       !TOOLS_WITH_CUSTOM_HANDLING.includes(
//                         toolCall.tool_name
//                       ) && (
//                         <ToolRunDisplay
//                           toolName={
//                             toolCall.tool_result && content
//                               ? `Used "${toolCall.tool_name}"`
//                               : `Using "${toolCall.tool_name}"`
//                           }
//                           toolLogo={
//                             <FiTool size={15} className="my-auto mr-1" />
//                           }
//                           isRunning={!toolCall.tool_result || !content}
//                         />
//                       )}

//                     {toolCall &&
//                       (!files || files.length == 0) &&
//                       toolCall.tool_name === IMAGE_GENERATION_TOOL_NAME &&
//                       !toolCall.tool_result && <GeneratingImageDisplay />}

//                     {toolCall &&
//                       toolCall.tool_name === INTERNET_SEARCH_TOOL_NAME && (
//                         <ToolRunDisplay
//                           toolName={
//                             toolCall.tool_result
//                               ? `Searched the internet`
//                               : `Searching the internet`
//                           }
//                           toolLogo={
//                             <FiGlobe size={15} className="my-auto mr-1" />
//                           }
//                           isRunning={!toolCall.tool_result}
//                         />
//                       )}

//                     {docs && docs.length > 0 && (
//                       <div className="mt-2 -mx-8 w-full mb-4 flex relative">
//                         <div className="w-full">
//                           <div className="px-8 flex gap-x-2">
//                             {!settings?.isMobile &&
//                               docs.length > 0 &&
//                               docs
//                                 .slice(0, 2)
//                                 .map((doc, ind) => (
//                                   <SourceCard
//                                     doc={doc}
//                                     key={ind}
//                                     setPresentingDocument={
//                                       setPresentingDocument
//                                     }
//                                   />
//                                 ))}
//                             <SeeMoreBlock
//                               documentSelectionToggled={
//                                 (documentSelectionToggled &&
//                                   selectedMessageForDocDisplay === messageId) ||
//                                 false
//                               }
//                               toggleDocumentSelection={toggleDocumentSelection}
//                               uniqueSources={uniqueSources}
//                             />
//                           </div>
//                         </div>
//                       </div>
//                     )}

//                     {content || files ? (
//                       <>
//                         <FileDisplay files={files || []} />

//                         {typeof content === "string" ? (
//                           <div className="overflow-x-visible max-w-content-max">
//                             {renderedMarkdown}
//                           </div>
//                         ) : (
//                           content
//                         )}
//                       </>
//                     ) : isComplete ? null : (
//                       <></>
//                     )}
//                   </div>

//                   {handleFeedback &&
//                     (isActive ? (
//                       <div
//                         className={`
//                         flex md:flex-row gap-x-0.5 mt-1
//                         transition-transform duration-300 ease-in-out
//                         transform opacity-100 translate-y-0"
//                   `}
//                       >
//                         <TooltipGroup>
//                           <div className="flex justify-start w-full gap-x-0.5">
//                             {includeMessageSwitcher && (
//                               <div className="-mx-1 mr-auto">
//                                 <MessageSwitcher
//                                   currentPage={currentMessageInd + 1}
//                                   totalPages={otherMessagesCanSwitchTo.length}
//                                   handlePrevious={() => {
//                                     onMessageSelection(
//                                       otherMessagesCanSwitchTo[
//                                         currentMessageInd - 1
//                                       ]
//                                     );
//                                   }}
//                                   handleNext={() => {
//                                     onMessageSelection(
//                                       otherMessagesCanSwitchTo[
//                                         currentMessageInd + 1
//                                       ]
//                                     );
//                                   }}
//                                 />
//                               </div>
//                             )}
//                           </div>
//                           <CustomTooltip showTick line content="Copy!">
//                             <CopyButton content={content.toString()} />
//                           </CustomTooltip>
//                           <CustomTooltip showTick line content="Good response!">
//                             <HoverableIcon
//                               icon={<LikeFeedback />}
//                               onClick={() => handleFeedback("like")}
//                             />
//                           </CustomTooltip>
//                           <CustomTooltip showTick line content="Bad response!">
//                             <HoverableIcon
//                               icon={<DislikeFeedback size={16} />}
//                               onClick={() => handleFeedback("dislike")}
//                             />
//                           </CustomTooltip>
//                           {regenerate && (
//                             <CustomTooltip
//                               disabled={isRegenerateDropdownVisible}
//                               showTick
//                               line
//                               content="Regenerate!"
//                             >
//                               <RegenerateOption
//                                 onDropdownVisibleChange={
//                                   setIsRegenerateDropdownVisible
//                                 }
//                                 onHoverChange={setIsRegenerateHovered}
//                                 selectedAssistant={currentPersona!}
//                                 regenerate={regenerate}
//                                 overriddenModel={overriddenModel}
//                               />
//                             </CustomTooltip>
//                           )}
//                         </TooltipGroup>
//                       </div>
//                     ) : (
//                       <div
//                         ref={hoverElementRef}
//                         className={`
//                         absolute -bottom-5
//                         z-10
//                         invisible ${
//                           (isHovering ||
//                             isRegenerateHovered ||
//                             settings?.isMobile) &&
//                           "!visible"
//                         }
//                         opacity-0 ${
//                           (isHovering ||
//                             isRegenerateHovered ||
//                             settings?.isMobile) &&
//                           "!opacity-100"
//                         }
//                         translate-y-2 ${
//                           (isHovering || settings?.isMobile) && "!translate-y-0"
//                         }
//                         transition-transform duration-300 ease-in-out
//                         flex md:flex-row gap-x-0.5 bg-background-125/40 -mx-1.5 p-1.5 rounded-lg
//                         `}
//                       >
//                         <TooltipGroup>
//                           <div className="flex justify-start w-full gap-x-0.5">
//                             {includeMessageSwitcher && (
//                               <div className="-mx-1 mr-auto">
//                                 <MessageSwitcher
//                                   currentPage={currentMessageInd + 1}
//                                   totalPages={otherMessagesCanSwitchTo.length}
//                                   handlePrevious={() => {
//                                     onMessageSelection(
//                                       otherMessagesCanSwitchTo[
//                                         currentMessageInd - 1
//                                       ]
//                                     );
//                                   }}
//                                   handleNext={() => {
//                                     onMessageSelection(
//                                       otherMessagesCanSwitchTo[
//                                         currentMessageInd + 1
//                                       ]
//                                     );
//                                   }}
//                                 />
//                               </div>
//                             )}
//                           </div>
//                           <CustomTooltip showTick line content="Copy!">
//                             <CopyButton content={content.toString()} />
//                           </CustomTooltip>

//                           <CustomTooltip showTick line content="Good response!">
//                             <HoverableIcon
//                               icon={<LikeFeedback />}
//                               onClick={() => handleFeedback("like")}
//                             />
//                           </CustomTooltip>

//                           <CustomTooltip showTick line content="Bad response!">
//                             <HoverableIcon
//                               icon={<DislikeFeedback size={16} />}
//                               onClick={() => handleFeedback("dislike")}
//                             />
//                           </CustomTooltip>
//                           {regenerate && (
//                             <CustomTooltip
//                               disabled={isRegenerateDropdownVisible}
//                               showTick
//                               line
//                               content="Regenerate!"
//                             >
//                               <RegenerateOption
//                                 selectedAssistant={currentPersona!}
//                                 onDropdownVisibleChange={
//                                   setIsRegenerateDropdownVisible
//                                 }
//                                 regenerate={regenerate}
//                                 overriddenModel={overriddenModel}
//                                 onHoverChange={setIsRegenerateHovered}
//                               />
//                             </CustomTooltip>
//                           )}
//                         </TooltipGroup>
//                       </div>
//                     ))}
//                 </div>
//               </div>
//             </div>
//           </div>
//         </div>
//         {(!toolCall || toolCall.tool_name === SEARCH_TOOL_NAME) &&
//           !query &&
//           continueGenerating && (
//             <ContinueGenerating handleContinueGenerating={continueGenerating} />
//           )}
//       </div>
//     </div>
//   );
// };

function MessageSwitcher({
  currentPage,
  totalPages,
  handlePrevious,
  handleNext,
}: {
  currentPage: number;
  totalPages: number;
  handlePrevious: () => void;
  handleNext: () => void;
}) {
  return (
    <div className="flex items-center text-sm space-x-0.5">
      <Hoverable
        icon={FiChevronLeft}
        onClick={currentPage === 1 ? undefined : handlePrevious}
      />

      <span className="text-emphasis select-none">
        {currentPage} / {totalPages}
      </span>

      <Hoverable
        icon={FiChevronRight}
        onClick={currentPage === totalPages ? undefined : handleNext}
      />
    </div>
  );
}

export const HumanMessage = ({
  content,
  files,
  messageId,
  otherMessagesCanSwitchTo,
  onEdit,
  onMessageSelection,
  shared,
  stopGenerating = () => null,
}: {
  shared?: boolean;
  content: string;
  files?: FileDescriptor[];
  messageId?: number | null;
  otherMessagesCanSwitchTo?: number[];
  onEdit?: (editedContent: string) => void;
  onMessageSelection?: (messageId: number) => void;
  stopGenerating?: () => void;
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [isHovered, setIsHovered] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(content);

  useEffect(() => {
    if (!isEditing) {
      setEditedContent(content);
    }
  }, [content, isEditing]);

  useEffect(() => {
    if (textareaRef.current) {
      // Focus the textarea
      textareaRef.current.focus();
      // Move the cursor to the end of the text
      textareaRef.current.selectionStart = textareaRef.current.value.length;
      textareaRef.current.selectionEnd = textareaRef.current.value.length;
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [isEditing]);

  const handleEditSubmit = () => {
    onEdit?.(editedContent);
    setIsEditing(false);
  };

  const currentMessageInd = messageId
    ? otherMessagesCanSwitchTo?.indexOf(messageId)
    : undefined;

  return (
    <div
      id="danswer-human-message"
      className="pt-5 pb-1 px-2 lg:px-5 flex -mr-6 relative"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className={`text-user-text mx-auto ${
          shared ? "w-full" : "w-[90%]"
        } max-w-[790px]`}
      >
        <div className="xl:ml-8">
          <div className="flex flex-col mr-4">
            <FileDisplay alignBubble files={files || []} />

            <div className="flex justify-end">
              <div className="w-full ml-8 flex w-full w-[800px] break-words">
                {isEditing ? (
                  <div className="w-full">
                    <div
                      className={`
                      opacity-100
                      w-full
                      flex
                      flex-col
                      border 
                      border-border 
                      rounded-lg 
                      bg-background-emphasis
                      pb-2
                      [&:has(textarea:focus)]::ring-1
                      [&:has(textarea:focus)]::ring-black
                    `}
                    >
                      <textarea
                        ref={textareaRef}
                        className={`
                        m-0 
                        w-full 
                        h-auto
                        shrink
                        border-0
                        rounded-lg 
                        overflow-y-hidden
                        bg-background-emphasis 
                        whitespace-normal 
                        break-word
                        overscroll-contain
                        outline-none 
                        placeholder-gray-400 
                        resize-none
                        pl-4
                        overflow-y-auto
                        pr-12 
                        py-4`}
                        aria-multiline
                        role="textarea"
                        value={editedContent}
                        style={{ scrollbarWidth: "thin" }}
                        onChange={(e) => {
                          setEditedContent(e.target.value);
                          textareaRef.current!.style.height = "auto";
                          e.target.style.height = `${e.target.scrollHeight}px`;
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Escape") {
                            e.preventDefault();
                            setEditedContent(content);
                            setIsEditing(false);
                          }
                          // Submit edit if "Command Enter" is pressed, like in ChatGPT
                          if (e.key === "Enter" && e.metaKey) {
                            handleEditSubmit();
                          }
                        }}
                      />
                      <div className="flex justify-end mt-2 gap-2 pr-4">
                        <button
                          className={`
                          w-fit
                          bg-accent 
                          text-inverted 
                          text-sm
                          rounded-lg 
                          inline-flex 
                          items-center 
                          justify-center 
                          flex-shrink-0 
                          font-medium 
                          min-h-[38px]
                          py-2
                          px-3
                          hover:bg-accent-hover
                        `}
                          onClick={handleEditSubmit}
                        >
                          Submit
                        </button>
                        <button
                          className={`
                          inline-flex 
                          items-center 
                          justify-center 
                          flex-shrink-0 
                          font-medium 
                          min-h-[38px] 
                          py-2 
                          px-3 
                          w-fit 
                          bg-hover
                          bg-background-strong 
                          text-sm
                          rounded-lg
                          hover:bg-hover-emphasis
                        `}
                          onClick={() => {
                            setEditedContent(content);
                            setIsEditing(false);
                          }}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                ) : typeof content === "string" ? (
                  <>
                    <div className="ml-auto mr-1 my-auto">
                      {onEdit &&
                      isHovered &&
                      !isEditing &&
                      (!files || files.length === 0) ? (
                        <TooltipProvider delayDuration={1000}>
                          <Tooltip>
                            <TooltipTrigger>
                              <button
                                className="hover:bg-hover p-1.5 rounded"
                                onClick={() => {
                                  setIsEditing(true);
                                  setIsHovered(false);
                                }}
                              >
                                <FiEdit2 className="!h-4 !w-4" />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent>Edit</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      ) : (
                        <div className="w-7" />
                      )}
                    </div>

                    <div
                      className={`${
                        !(
                          onEdit &&
                          isHovered &&
                          !isEditing &&
                          (!files || files.length === 0)
                        ) && "ml-auto"
                      } relative flex-none max-w-[70%] mb-auto whitespace-break-spaces rounded-3xl bg-user px-5 py-2.5`}
                    >
                      {content}
                    </div>
                  </>
                ) : (
                  <>
                    {onEdit &&
                    isHovered &&
                    !isEditing &&
                    (!files || files.length === 0) ? (
                      <div className="my-auto">
                        <Hoverable
                          icon={FiEdit2}
                          onClick={() => {
                            setIsEditing(true);
                            setIsHovered(false);
                          }}
                        />
                      </div>
                    ) : (
                      <div className="h-[27px]" />
                    )}
                    <div className="ml-auto rounded-lg p-1">{content}</div>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-col md:flex-row gap-x-0.5 mt-1">
            {currentMessageInd !== undefined &&
              onMessageSelection &&
              otherMessagesCanSwitchTo &&
              otherMessagesCanSwitchTo.length > 1 && (
                <div className="ml-auto mr-3">
                  <MessageSwitcher
                    currentPage={currentMessageInd + 1}
                    totalPages={otherMessagesCanSwitchTo.length}
                    handlePrevious={() => {
                      stopGenerating();
                      onMessageSelection(
                        otherMessagesCanSwitchTo[currentMessageInd - 1]
                      );
                    }}
                    handleNext={() => {
                      stopGenerating();
                      onMessageSelection(
                        otherMessagesCanSwitchTo[currentMessageInd + 1]
                      );
                    }}
                  />
                </div>
              )}
          </div>
        </div>
      </div>
    </div>
  );
};
