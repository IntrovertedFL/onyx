import { redirect } from "next/navigation";
import { unstable_noStore as noStore } from "next/cache";
import { InstantSSRAutoRefresh } from "@/components/SSRAutoRefresh";
import { WelcomeModal } from "@/components/initialSetup/welcome/WelcomeModalWrapper";
import { ChatProvider } from "@/components/context/ChatContext";
import { fetchChatData } from "@/lib/chat/fetchChatData";
import WrappedChat from "./WrappedChat";
import { cookies } from "next/headers";

export default async function Page(props: {
  searchParams: Promise<{ [key: string]: string }>;
}) {
  const searchParams = await props.searchParams;
  const firstMessage = searchParams.firstMessage;
  // noStore();
  // const requestCookies = await cookies();
  // const data = await fetchChatData(searchParams);
  // const defaultSidebarOff = searchParams.defaultSidebarOff === "true";

  // if ("redirect" in data) {
  //   redirect(data.redirect);
  // }

  // const {
  //   user,
  //   chatSessions,
  //   availableSources,
  //   documentSets,
  //   tags,
  //   llmProviders,
  //   folders,
  //   toggleSidebar,
  //   openedFolders,
  //   defaultAssistantId,
  //   shouldShowWelcomeModal,
  //   ccPairs,
  // } = data;

  // return (
  //   <>
  //     <InstantSSRAutoRefresh />
  //     {shouldShowWelcomeModal && (
  //       <WelcomeModal user={user} requestCookies={requestCookies} />
  //     )}
  //     <ChatProvider
  //       value={{
  //         chatSessions,
  //         availableSources,
  //         ccPairs,
  //         documentSets,
  //         tags,
  //         availableDocumentSets: documentSets,
  //         availableTags: tags,
  //         llmProviders,
  //         folders,
  //         openedFolders,
  //         shouldShowWelcomeModal,
  //         defaultAssistantId,
  //       }}
  return <WrappedChat firstMessage={firstMessage} initiallyToggled={false} />;
  //   </ChatProvider>
  // </>
  // );
}
