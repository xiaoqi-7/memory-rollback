"""Prompt templates for :class:`agent.react_agent.ReactAgent`.

Each REACT sub-step (claim, plan, respond, reconcile memory) builds a system
and user message. The builder functions below own the prompt *text*; the agent
supplies the dynamic blocks (domain prompt, retrieved memories, tool specs,
observations, etc.). Keeping the wording here lets the prompts be edited and
reviewed without touching the control flow in ``react_agent.py``.
"""

# --------------------------------------------------------------------------- #
# make_claim
# --------------------------------------------------------------------------- #
def claim_system(domain_prompt: str) -> str:
    return (
        f"{domain_prompt}\n\n"
        "Read the user's request, the relevant memories, and the prior "
        "user turns, then state a single concise claim describing exactly "
        "what the user wants now. In one \"used\" list, cite every memory id "
        "(e.g. m_001) and prior-turn id you relied on. Each prior turn is "
        "shown in brackets with its own id (e.g. user_input_2) at the start "
        "of its line — cite that exact id. The current user request may also "
        "be shown with a bracketed id (e.g. user_input_4) next to \"User "
        "request\"; cite that id too when the claim relied directly on the "
        "user's current wording (optional, only when it did).\n"
        'Respond ONLY as JSON: {"claim": "<text>", "used": '
        '["m_001", "user_input_2", ...]}'
    )


def claim_user(
    prompt: str, memory_block: str, history_block: str, user_input_id: str = ""
) -> str:
    request_line = (
        f"User request [{user_input_id}]:\n{prompt}" if user_input_id
        else f"User request:\n{prompt}"
    )
    return (
        f"{request_line}\n\n"
        f"Relevant memories:\n{memory_block}\n\n"
        f"Prior user turns:\n{history_block}"
    )


# --------------------------------------------------------------------------- #
# plan_actions
# --------------------------------------------------------------------------- #
def plan_system(
    domain_prompt: str,
    guidance_block: str,
    tool_specs: str,
    memory_block: str,
    obs_block: str,
    error_block: str,
) -> str:
    return (
        f"{domain_prompt}\n\n"
        "You plan tool calls to fulfil the user's request. Only use the "
        "tools listed below, with valid argument names.\n\n"
        "Tool-calling policy (follow exactly):\n"
        "- Tools come in two kinds. READ-ONLY tools (search, lookup, and "
        "checks for price, shipping, or policy) only fetch information and "
        "have no side effects. SIDE-EFFECTING tools (open/create a ticket, "
        "book, reserve, cancel — anything that creates or changes a record) "
        "take a real action the user has to live with. The rules below "
        "differ for the two kinds.\n"
        "- READ-ONLY: if the request maps to a read-only tool, you MUST plan "
        "that call. Never skip a relevant read-only tool because you guess "
        "the result will be unhelpful or unnecessary, and never tell the "
        "user you cannot retrieve information when a read-only tool exists. "
        "When in doubt about a read-only tool, call it.\n"
        "- SIDE-EFFECTING: plan one ONLY when the user's current message is "
        "an explicit, affirmative instruction to take that action now (e.g. "
        '"open a ticket", "please file it", "go ahead and book"). Do NOT '
        "plan a side-effecting call when the user is only asking whether to "
        '("could we open a ticket?", "should I?"), floating or considering '
        'it ("I might want a refund", "I prefer a replacement"), clarifying '
        'or correcting a detail ("actually the issue is the shoes, not the '
        'headphones"), or retracting or disregarding something ("disregard '
        'the smartwatch issue", "forget the ticket"). In those cases plan no '
        "side-effecting call — the answer will advise the option instead. "
        "When unsure whether the user truly wants the action taken now, do "
        "NOT take it; advise instead.\n"
        "- When a single message contains both a retraction and a new "
        "affirmative instruction (e.g. 'forget X, just do Y instead' or "
        "'never mind X, process Y'), plan NO tool for the retracted item "
        "and plan the appropriate tool for the new instruction. Phrases "
        "like 'just process', 'go ahead with', 'please do', 'schedule', "
        "'proceed with' followed by an action name are explicit imperatives "
        "even when combined with a retraction in the same message.\n"
        "- Calling a search or lookup tool is always allowed, even for "
        "actions you cannot truly finalize (e.g. booking or payment): plan "
        "the search/lookup so you can report what is available.\n"
        "- Fill argument values from the relevant memories and prior tool "
        "results below when the user established a detail earlier (e.g. a "
        "destination, airline, order id, or product id) without repeating "
        "it now. Do not ask the user to repeat what is already known.\n"
        "- Chain calls when one tool's output supplies another's arguments "
        "(e.g. use a product id returned by a search to then check its "
        "shipping or price). If the argument you need is not yet in the "
        "prior tool results or memories, plan the call that produces it "
        "first; it will be available when you plan again.\n"
        "- Id-type arguments (product id, order id, flight id, etc.) must "
        "be a real id returned by a search or lookup tool or stored in the "
        "memories below — never invent one from a human-readable name. If "
        "you only know the name, plan the search/lookup that resolves its "
        "id first, then chain the call that needs the id.\n"
        "- If a prior call below failed (see failed calls), do not repeat it "
        "unchanged. Diagnose the cause and fix it: a 'not found' error on an "
        "id argument usually means you passed a name instead of a real id — "
        "plan the search/lookup that resolves the id, then retry the call "
        "with the resolved id.\n"
        "- MEMORY-FIRST: before planning any tool call, check whether the "
        "relevant memories below already answer the request in full. If the "
        "memories contain everything needed to respond (a stable user "
        "preference, constraint, identifier, or a fact established earlier "
        "this session that directly and completely answers what the user is "
        "asking now), plan NO tool calls — return an empty steps list and "
        "let the response be generated from memory. Only fall through to "
        "planning a tool when memory does not fully cover the request. Do "
        "not call a read-only tool merely to rediscover or re-confirm a "
        "stable fact memory already holds.\n"
        "- Use tools when the needed information is external, volatile, unavailable"
        "in memory, or must be verified against the simulator/database, such as "
        "current availability, current price, shipping status, refund policy, "
        "flight/hotel search results, or order status."
        "- If memory already contains a real id needed by a tool call, use that "
        "id directly. Do not search again only to resolve the id."
        "- If prior tool results already contain the needed result for the same "
        "arguments, do not repeat the call."
        "- If the user has updated an attribute (size, color, destination, "
        "cabin, etc.) earlier in this session and the current request asks "
        "for information about that item, you MUST search using the UPDATED "
        "attribute value, not the original one. Check the relevant memories "
        "for the most recent value of each attribute before constructing "
        "tool call arguments. Never use a stale attribute value from an "
        "earlier memory if a more recent memory supersedes it.\n"
        "- When the user issues a clear imperative to take a side-effecting "
        "action NOW (open, file, create, raise, book, reserve, cancel, "
        "etc.), plan the corresponding tool call even if a memory or prior "
        "result shows a similar action was already taken — the user is "
        "explicitly asking again, so it is not your job to decide the repeat "
        "is redundant. But this applies only to a genuine imperative this "
        "turn: a question about whether to act, a tentative or considered "
        "option, a clarification or correction of an earlier detail, and a "
        "retraction or 'disregard'/'forget' request are NOT imperatives — do "
        "not plan a side-effecting call for any of those.\n"
        "- Do not re-plan a call already present in the prior tool results "
        "with the same arguments. Within a single response, plan each "
        "distinct call only once — never list the same tool with the same "
        "intent twice."
        f"{guidance_block}\n"
        f"Available tools:\n{tool_specs}\n\n"
        "Relevant memories (use for argument values):\n"
        f"{memory_block}\n\n"
        "Tool results so far this turn (chain off these; do not repeat "
        f"them):\n{obs_block}\n\n"
        "Failed calls so far this turn (fix the cause; do not repeat them "
        f"unchanged):\n{error_block}\n\n"
        "In one \"used\" list, cite every memory id (e.g. m_001), prior-turn "
        "id (each prior turn is shown in brackets with its own id, e.g. "
        "user_input_2), and the claim step id this plan relied on. The claim "
        "step id is the value shown in brackets next to \"Claim\" in the user "
        "message below (e.g. s_002) — write that exact id into \"used\" if "
        "this plan relied on the claim. The current user request may also be "
        "shown with a bracketed id (e.g. user_input_4) next to \"User "
        "request\"; cite that id too when the plan relied directly on the "
        "user's current wording (optional, only when it did, not just "
        "transitively via the claim).\n"
        'Respond ONLY as JSON: {"steps": [{"type": "tool", "tool_name": '
        '"<name>", "args": {<arg>: <value>}}], "used": ["m_001", '
        '"user_input_2", "<claim_step_id>", ...]}. Plan the calls that are '
        "still needed; use an empty list only when every needed call is "
        "already in the prior tool results."
    )


def plan_user(
    prompt: str, claim_text: str, claim_step_id: str = "", user_input_id: str = ""
) -> str:
    request_line = (
        f"User request [{user_input_id}]:\n{prompt}" if user_input_id
        else f"User request:\n{prompt}"
    )
    claim_block = (
        f"Claim [{claim_step_id}]:\n{claim_text}" if claim_step_id
        else f"Claim:\n{claim_text}"
    )
    return f"{request_line}\n\n{claim_block}"


def plan_guidance_block(tool_guidance: str) -> str:
    return (
        f"\nWhich requests map to which tools:\n{tool_guidance}\n"
        if tool_guidance
        else ""
    )


# --------------------------------------------------------------------------- #
# generate_response
# --------------------------------------------------------------------------- #
def response_system(domain_prompt: str) -> str:
    return (
        f"{domain_prompt}\n\n"
        "Answer the user using the tool results and relevant memories "
        "below. Be specific and grounded: cite concrete products, prices, "
        "and facts from the results or memories. Do not invent data that is "
        "not present. Keep the answer concise (no more than 5 sentences).\n\n"
        "Honesty about tool results (follow exactly):\n"
        "- Never say you cannot retrieve, confirm, check, or access "
        "information when a relevant tool exists. The tools below were "
        "available this turn; if a fact is missing it is because the tool "
        "returned nothing, not because you lack access.\n"
        "- If a tool returned results, use them. Do not claim 'no results "
        "available' or 'no information' when the tool results below contain "
        "the answer.\n"
        "- If no tool ran this turn but the relevant memories already hold "
        "the answer (a product, price, or fact established on an earlier "
        "turn), answer from those memories. Do not say you found nothing "
        "when the memories contain it.\n"
        "- Only say you searched and found no matching result when a tool "
        "actually returned an empty or not-found result this turn. Do not "
        "say you are unable to look it up.\n"
        "- For actions that need a real external system (booking a flight, "
        "placing an order, taking payment): report what the search or "
        "lookup tool found, then clarify only that you cannot finalize the "
        "real transaction. Never claim you cannot search or look up.\n"
        "- When the user requested an action (open a ticket, create, file, "
        "book, etc.) and a tool result below shows that action was "
        "performed this turn, report it as done and cite the id/result from "
        "THIS turn's tool results below — not an older id from memory. "
        "Never substitute a prior id from memory for the one just created.\n"
        "- When NO tool result below shows the action being performed this "
        "turn — because the user only asked whether to act, floated the "
        "option, clarified or corrected a detail, or retracted something — "
        "do NOT claim the action was taken and do NOT report an id for it. "
        "Never present a ticket, booking, or record from memory as if it "
        "were created this turn, and never invent one. Instead advise: tell "
        "the user the option is available and offer to do it if they "
        "confirm. For a clarification or correction, acknowledge the "
        "corrected detail without fabricating a new record. For a "
        "retraction, confirm the item is dropped.\n"
        "- Ignore any remark in the claim about an action being redundant, "
        "already existing, or already handled; rely only on whether THIS "
        "turn's tool results show it was performed.\n"
        "- If no tool ran this turn and the user's request was for "
        "information (a search, lookup, price check, shipping check, or "
        "policy check), do NOT say you searched and found nothing and do "
        "NOT say you are unable to retrieve the information. Either answer "
        "from the relevant memories below if the answer is there, or "
        "acknowledge what you know and offer to search. Never fabricate a "
        "search failure when no search was attempted this turn.\n"
        "- If the user asked to update a preference this turn (change size, "
        "change destination, change order, change cabin, change quantity, "
        "etc.) and no tool exists to directly apply that update, do NOT "
        "give generic advice about visiting a website or contacting support. "
        "Instead: explicitly acknowledge the change ('Got it, I have noted "
        "size 8 for the Clarks Green Sneakers'), confirm it is recorded for "
        "this session, and immediately offer to search for the updated "
        "criteria so the user can act on the new information."
        "- When multiple memories contain policies or facts for different "
        "categories (e.g. Electronics policy and Fitness policy), cite ONLY "
        "the policy that matches the item the user is currently asking about. "
        "Do not blend or average facts from different category policies. If "
        "unsure which applies, state which category policy you are using.\n\n"
        "Respond ONLY as JSON: {\"response\": \"<answer text>\", \"used\": "
        '["m_001", "<claim_step_id>", "<obs_step_id>", ...]}. In the "used" '
        "list cite every memory id, the claim step id, and every "
        "tool_observation step id this turn the answer drew on. The claim "
        "step id is shown in brackets next to \"Claim\" and each "
        "tool_observation step id is shown in brackets at the start of its "
        "line under \"Tool results\" — write those exact ids into \"used\". "
        "The current user request may also be shown with a bracketed id (e.g. "
        "user_input_4) next to \"User request\"; cite that id too when the "
        "answer relied directly on the user's current wording (optional, only "
        "when it did, not just transitively via the claim). "
        'The "response" value must obey every instruction above (the '
        "5-sentence cap, grounding, and honesty-about-tool-results rules)."
    )


def response_user(
    prompt: str, claim_text: str, memory_block: str, obs_block: str,
    claim_step_id: str = "", user_input_id: str = "",
) -> str:
    request_line = (
        f"User request [{user_input_id}]:\n{prompt}" if user_input_id
        else f"User request:\n{prompt}"
    )
    claim_block = (
        f"Claim [{claim_step_id}]:\n{claim_text}" if claim_step_id
        else f"Claim about intent:\n{claim_text}"
    )
    return (
        f"{request_line}\n\n"
        f"{claim_block}\n\n"
        f"Relevant memories:\n{memory_block}\n\n"
        f"Tool results:\n{obs_block}"
    )


# --------------------------------------------------------------------------- #
# extract_memories
# --------------------------------------------------------------------------- #
def memory_guidance_block(memory_guidance: str) -> str:
    return (
        f"\nDomain guidance on what is worth remembering:\n{memory_guidance}\n"
        if memory_guidance
        else ""
    )


def extract_system(domain_prompt: str, guidance_block: str) -> str:
    return (
        f"{domain_prompt}\n\n"
        "Maintain the session's durable memory after this turn. Consider "
        "the existing memories, the current turn, and the prior user turns "
        "the agent relied on, then decide four things.\n"
        "ATOMICITY PRINCIPLE (applies to every action below): each memory "
        "holds ONE atomic fact about ONE subject — a single product, a single "
        "preference, a single constraint, a single action/record, a single "
        "policy. NEVER collapse facts about different subjects into one entry. "
        "Keep distinct products, preferences, orders, tickets, and policies as "
        "SEPARATE memories. Do not over-summarize: the store should end the "
        "session with one active memory per distinct fact, not a single "
        "merged paragraph. A product fact may carry that product's own "
        "attributes (id, name, category, color, size, price, stock) together "
        "because they describe the one product — but two different products, "
        "or a product and a separate preference, must stay apart.\n"
        "TOOL-ANCHORED FACTS (ABSOLUTE RULE): a memory sourced from "
        "\"tool_observation\" (a price, shipping quote, availability check, "
        "or order confirmation) describes something independently true — a "
        "record of what was found, not a statement of the user's current "
        "intent. It NEVER becomes false, stale, or redundant because of "
        "anything the user says. You MUST NOT touch a tool_observation "
        "memory under any circumstance: never update it, never delete it, "
        "never include it in a consolidation, and never overwrite, refine, "
        "or rephrase it — no matter what the user retracts, changes, "
        "corrects, or repeats. A tool_observation memory_id must NEVER "
        "appear in the updates, deletes, or consolidations lists. The only "
        "valid response to a user changing their mind about a tool-anchored "
        "fact is to add a NEW preference/intent memory (item 1) capturing "
        "the new stance and leave every tool_observation memory exactly as "
        "it is. This absolutely governs items 2, 3, and 4 below.\n"
        "Likewise, when producing updates (item 2) or consolidations "
        "(item 4), NEVER set the source of the resulting memory to "
        "\"tool_observation\" and NEVER draw the corrected/merged content "
        "from a tool_observation memory. tool_observation is only ever a "
        "source for brand-new facts (item 1) grounded in a tool that ran "
        "THIS turn.\n"
        "1. new: facts derived from the user this turn that are NOT already "
        "stored. Before adding any fact to \"new\", scan the existing "
        "memories: if one already covers the SAME subject (the same product, "
        "preference, constraint, order, or policy), do NOT create a new "
        "memory. If the user merely restated it with no change, leave the "
        "existing memory as-is (add nothing). If the user changed or refined "
        "it, route it to \"updates\" (item 2) on that existing memory_id — "
        "never record the same or refined fact as a new entry alongside the "
        "old one. A fact you put in \"updates\" must NOT also appear in "
        "\"new\": choose exactly one. Refining or restating an existing fact "
        "is ONLY an update (or nothing) — it is never both an update and a "
        "new entry. "
        "Only add to \"new\" when no existing memory covers that "
        "subject. Record EVERY distinct piece of information the user "
        "reveals or implies: stated preferences, constraints, attributes, "
        "decisions, the products or categories they ask about, the brand, "
        "color, size, fit, material, price or budget, quantity, and "
        "shipping or address details they mention, and any option they say "
        "they are considering or rejecting. Capture it the moment it "
        "appears, even from a single passing request not yet committed to. "
        "MANDATORY TOOL CAPTURE: whenever ANY tool was called this turn (the "
        "tool results below are non-empty), you MUST add a \"new\" "
        "tool_observation memory for EACH distinct tool result, capturing its "
        "content verbatim-faithfully (ids, names, prices, stock, shipping, "
        "policy text, confirmations). This is required regardless of whether "
        "the user adopted, acted on, or even reacted to the result — the fact "
        "that the tool returned it is enough. Do NOT skip a tool result "
        "because it seems unhelpful, the user did not commit to it, or you "
        "doubt it will be reused later. The only tool result you may omit is "
        "one already stored verbatim in an existing tool_observation memory "
        "with the same ids/values. "
        "Also record facts the user has effectively adopted from the tool "
        "results (e.g. a specific product they decided on or asked to act "
        "on). Prefer recording a fact over dropping it; only skip pure "
        "chit-chat that carries no information about the user. ALWAYS record "
        "a user preference, constraint, or decision the moment it appears — "
        "these are what let later turns answer from memory without a "
        "redundant tool call, so never let a stated preference go "
        "unrecorded."
        f"{guidance_block} State "
        "each as a concise standalone fact and do not copy a turn verbatim. "
        "Choose source by where the fact comes from: \"user_history\" for "
        "facts distilled from the prior user turns, \"tool_observation\" for "
        "facts grounded in any tool result this turn (whether or not the user "
        "acted on it), "
        "\"user_input\" for facts stated in the current user request, and "
        "\"agent_summary\" for facts the agent synthesized.\n"
        "Only use \"tool_observation\" when a tool was actually called THIS "
        "turn and the fact is grounded in that result. If no tool ran this "
        "turn, never use \"tool_observation\" — use \"user_input\" for what "
        "the user just said, \"user_history\" for facts drawn from earlier "
        "turns, or \"agent_summary\" for something you synthesized. "
        "Mislabeling a fact as tool_observation when no tool ran makes it "
        "incorrectly immune to later updates and deletes.\n"
        "When a fact is grounded in a tool result (source "
        "\"tool_observation\"), preserve every stable identifier the tool "
        "returned verbatim inside the fact text — product_id, order_id, "
        "flight_id, hotel_id, ticket_id. NEVER store only a human-readable "
        "name when the tool also returned an id: the id is what later turns "
        "reuse to call tools without re-searching, so a name-only memory "
        "forces a wasteful repeat lookup. Lead with the id so it is easy to "
        "find. For a shopping product, record the product_id, name, category, "
        "color, size, price, and stock status together in one fact, e.g. "
        "\"Product P0164 is Clarks Green Sneakers, category Footwear, color "
        "Green, size 6, price $129.95, in stock [source: search_product "
        "query='Clarks Green Sneakers'].\" Close the fact with a short "
        "provenance tag in square brackets naming the tool and its key "
        "argument so the fact's origin stays traceable. If the tool result "
        "is missing one of these fields, simply omit that field — still "
        "record the fact with the fields you do have, and never invent a "
        "value to fill a gap.\n"
        "2. updates: existing memories whose content is now outdated or "
        "should be refined. Whenever the user changes, corrects, or "
        "overrides a detail they stated earlier (e.g. \"actually make it one "
        "person\" after saying two, \"change the destination to Sydney\", "
        "\"the issue is the shoes, not the headphones\", \"change the size "
        "to 8\"), you MUST update the memory that held the old value to the "
        "new value — do not leave the stale fact active and do not record "
        "the correction as a brand-new fact alongside the old one. "
        "Give the memory_id and the corrected content. "
        "CRITICAL: when the user changes a specific attribute (size, color, "
        "destination, order ID, airline, cabin, quantity) for an item "
        "already in memory, always UPDATE the existing memory for that item "
        "rather than creating a new one. If both an old and a new value "
        "exist as separate active memories for the same item, the agent "
        "will retrieve both and use the wrong one. The old value must be "
        "superseded so it is no longer active.\n"
        "EXCEPTION — see TOOL-ANCHORED FACTS above. NEVER update or "
        "overwrite a tool_observation memory, and never base the corrected "
        "content on one. If the old value lives in a tool_observation "
        "memory, leave it untouched and add the corrected value as a new "
        "preference memory (item 1) instead. Only update in place when the "
        "old value was a non-tool fact that was simply a misstatement "
        "nothing else depends on.\n"
        "3. deletes: existing memories that are now contradicted or no "
        "longer valid. Whenever the user retracts, withdraws, cancels, or "
        "tells you to disregard or forget something they raised earlier "
        "(e.g. \"disregard the smartwatch issue\", \"forget the hotel\", "
        "\"I don't need that anymore\"), you MUST delete the memory holding "
        "that item or intent. Give the memory_id.\n"
        "CRITICAL — delete ONLY the memory that matches the retracted "
        "subject. Read each candidate memory's content and delete it only if "
        "its subject is the very thing the user is dropping. \"Forget the "
        "shipping to Melbourne\" retracts the SHIPPING/DELIVERY fact for "
        "Melbourne — delete that memory, and leave the product, price, "
        "preference, and any other still-valid memories untouched. Retracting "
        "one attribute or sub-fact (shipping, a delivery estimate, a "
        "competitor price) does NOT mean delete the whole product the user "
        "still cares about. If the retraction targets only part of a memory "
        "that also holds still-valid facts, do NOT delete the whole memory: "
        "instead update it (item 2) to remove only the retracted part. Never "
        "delete a memory whose subject the user did not mention. When no "
        "existing memory clearly matches the retracted subject, delete "
        "nothing.\n"
        "EXCEPTION — see TOOL-ANCHORED FACTS above. NEVER delete a "
        "tool_observation memory; add a new preference memory capturing the "
        "retraction instead, and leave the original untouched.\n"
        "4. consolidations: groups of two or more existing memories that are "
        "TRUE DUPLICATES of the SAME atomic fact about the SAME subject — i.e. "
        "they restate the same product, the same preference, or the same "
        "constraint with no additional distinct information. Merge ONLY these "
        "into a single entry. Give the list of source memory_ids and the "
        "merged content. Do NOT consolidate memories that describe different "
        "subjects (two different products, a product and a preference, two "
        "different policies) just because they are related or were mentioned "
        "together — keeping them separate is required by the atomicity "
        "principle above. When in doubt, do NOT consolidate; leave the "
        "memories as distinct active facts.\n"
        "A tool_observation memory must NEVER be one of the memories in a "
        "consolidation — see TOOL-ANCHORED FACTS above. Exclude every "
        "tool_observation memory_id from the consolidations list and leave "
        "it as its own distinct active fact; only ever merge non-tool "
        "duplicate memories with each other.\n"
        "Only reference memory_ids that appear in the existing memories. "
        "Do not reference the same memory in more than one of updates, "
        "deletes, or consolidations.\n"
        "PROVENANCE — every entry in \"new\", \"updates\", "
        "\"consolidations\", and \"deletes\" MUST carry a \"used\" list citing "
        "what the entry is grounded in: every existing memory id (e.g. m_001), "
        "prior-turn id (each relied-on prior turn is shown in brackets with its "
        "own id, e.g. user_input_2), the current user request id (shown in "
        "brackets next to \"User request\", e.g. user_input_4), the claim step "
        "id, and/or tool_observation step id it drew on. Cite the claim step id "
        "whenever the fact came from the user's intent this turn, the "
        "tool_observation step id whenever the fact came from a tool result "
        "this turn, and the current user request id when a step relied directly "
        "on the user's current wording (optional, only when it did). For "
        "\"deletes\", cite every memory id, prior-turn id, current user request "
        "id, claim step id, and/or tool_observation step id the deletion "
        "decision drew on, exactly as for new/updates/consolidations — e.g. the "
        "user's current message saying \"forget X\" -> cite the current user "
        "request id; a tool result proving the old value wrong -> cite that "
        "tool_observation step id; a contradicting existing memory -> cite that "
        "memory id. A delete naming only a memory_id with an empty \"used\" "
        "loses the record of WHY it happened, so always justify it.\n"
        'Respond ONLY as JSON: {"new": [{"content": "<fact>", "source": '
        '"user_input|user_history|tool_observation|agent_summary", '
        '"trust_score": <0-1>, "used": ["m_001", "user_input_2", '
        '"<claim_step_id>", "<obs_step_id>", ...]}], "updates": '
        '[{"memory_id": "m_001", '
        '"content": "<fact>", "trust_score": <0-1>, "used": ["m_001", '
        '"<claim_step_id>", ...]}], "deletes": '
        '[{"memory_id": "m_002", "used": ["user_input_3", ...]}], '
        '"consolidations": [{"memory_ids": '
        '["m_003", "m_004"], "content": "<merged fact>", "trust_score": '
        '<0-1>, "used": ["m_003", "m_004", "<claim_step_id>", ...]}]}'
    )


def extract_user(
    prompt: str,
    response: str,
    history_block: str,
    obs_block: str,
    existing_block: str,
    claim_block: str,
    user_input_id: str = "",
) -> str:
    request_line = (
        f"User request [{user_input_id}]:\n{prompt}" if user_input_id
        else f"User request:\n{prompt}"
    )
    return (
        f"{request_line}\n\n"
        f"Claim about intent:\n{claim_block}\n\n"
        # f"Agent response:\n{response}\n\n"
        f"Prior user turns relied on:\n{history_block}\n\n"
        f"Tool results this turn:\n{obs_block}\n\n"
        f"Existing memories:\n{existing_block}"
    )
