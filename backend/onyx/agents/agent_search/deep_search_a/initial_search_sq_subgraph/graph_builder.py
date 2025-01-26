from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from onyx.agents.agent_search.deep_search_a.answer_initial_sub_question.graph_builder import (
    answer_query_graph_builder,
)
from onyx.agents.agent_search.deep_search_a.base_raw_search.graph_builder import (
    base_raw_search_graph_builder,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.edges import (
    parallelize_initial_sub_question_answering,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.nodes.generate_initial_answer import (
    generate_initial_answer,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.nodes.ingest_initial_base_retrieval import (
    ingest_initial_base_retrieval,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.nodes.ingest_initial_sub_question_answers import (
    ingest_initial_sub_question_answers,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.nodes.initial_answer_quality_check import (
    initial_answer_quality_check,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.nodes.initial_sub_question_creation import (
    initial_sub_question_creation,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.nodes.retrieval_consolidation import (
    retrieval_consolidation,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.states import (
    SearchSQInput,
)
from onyx.agents.agent_search.deep_search_a.initial_search_sq_subgraph.states import (
    SearchSQState,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

test_mode = False


def initial_search_sq_subgraph_builder(test_mode: bool = False) -> StateGraph:
    graph = StateGraph(
        state_schema=SearchSQState,
        input=SearchSQInput,
    )

    graph.add_node(
        node="initial_sub_question_creation",
        action=initial_sub_question_creation,
    )
    answer_query_subgraph = answer_query_graph_builder().compile()
    graph.add_node(
        node="answer_query_subgraph",
        action=answer_query_subgraph,
    )

    base_raw_search_subgraph = base_raw_search_graph_builder().compile()
    graph.add_node(
        node="base_raw_search_subgraph",
        action=base_raw_search_subgraph,
    )

    graph.add_node(
        node="ingest_initial_retrieval",
        action=ingest_initial_base_retrieval,
    )

    graph.add_node(
        node="retrieval_consolidation",
        action=retrieval_consolidation,
    )

    graph.add_node(
        node="ingest_initial_sub_question_answers",
        action=ingest_initial_sub_question_answers,
    )
    graph.add_node(
        node="generate_initial_answer",
        action=generate_initial_answer,
    )

    graph.add_node(
        node="initial_answer_quality_check",
        action=initial_answer_quality_check,
    )

    ### Add edges ###

    # raph.add_edge(start_key=START, end_key="base_raw_search_subgraph")

    graph.add_edge(
        start_key=START,
        end_key="base_raw_search_subgraph",
    )

    # graph.add_edge(
    #     start_key="agent_search_start",
    #     end_key="entity_term_extraction_llm",
    # )

    graph.add_edge(
        start_key=START,
        end_key="initial_sub_question_creation",
    )

    graph.add_edge(
        start_key="base_raw_search_subgraph",
        end_key="ingest_initial_retrieval",
    )

    graph.add_edge(
        start_key=["ingest_initial_retrieval", "ingest_initial_sub_question_answers"],
        end_key="retrieval_consolidation",
    )

    graph.add_edge(
        start_key="retrieval_consolidation",
        end_key="generate_initial_answer",
    )

    # graph.add_edge(
    #     start_key="LLM",
    #     end_key=END,
    # )

    # graph.add_edge(
    #     start_key=START,
    #     end_key="initial_sub_question_creation",
    # )

    graph.add_conditional_edges(
        source="initial_sub_question_creation",
        path=parallelize_initial_sub_question_answering,
        path_map=["answer_query_subgraph"],
    )
    graph.add_edge(
        start_key="answer_query_subgraph",
        end_key="ingest_initial_sub_question_answers",
    )

    graph.add_edge(
        start_key="retrieval_consolidation",
        end_key="generate_initial_answer",
    )

    graph.add_edge(
        start_key="generate_initial_answer",
        end_key="initial_answer_quality_check",
    )

    graph.add_edge(
        start_key="initial_answer_quality_check",
        end_key=END,
    )

    # graph.add_edge(
    #     start_key="generate_refined_answer",
    #     end_key="check_refined_answer",
    # )

    # graph.add_edge(
    #     start_key="check_refined_answer",
    #     end_key=END,
    # )

    return graph
