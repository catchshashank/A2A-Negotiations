# ============================================================
# Conversation.py
# ============================================================
# Manages one buyer-vs-seller negotiation session.
#
# Faithfully replicates the original A2A-NT Conversation.py.
# Internal logic is unchanged; comments are added for clarity.
# ============================================================

import json
import os
import re
import logging
logger = logging.getLogger(__name__)

from core.LanguageModel import LanguageModel


class Conversation:
    """
    Runs a single bilateral negotiation between a buyer agent and
    a seller agent, both powered by LLMs.

    A third "summary" model is used as a neutral judge to:
      - Extract the current price offer from seller messages
      - Decide whether the negotiation has concluded
    """

    def __init__(
        self,
        product_data: dict,
        buyer_model: str = "gpt-3.5-turbo",
        seller_model: str = "gpt-3.5-turbo",
        summary_model: str = "gpt-3.5-turbo",
        max_turns: int = 30,
        experiment_num: int = 0,
        budget: float = None,
    ):
        self.product_data = product_data
        self.buyer_model_name = buyer_model
        self.seller_model_name = seller_model
        self.summary_model_name = summary_model

        self.buyer_model   = LanguageModel(model_name=buyer_model)
        self.seller_model  = LanguageModel(model_name=seller_model)
        self.summary_model = LanguageModel(model_name=summary_model)

        self.conversation_history: list = []
        self.max_turns = max_turns
        self.completed_turns = 0
        self.experiment_num = experiment_num
        self.budget = budget
        self.budget_scenario: str = None

        self.product_id = product_data.get("id", 0)

        # Price tracking — first element is always the retail price
        retail_price_str = product_data["Retail Price"]
        self.seller_price_offers: list = [
            float(retail_price_str.replace("$", "").replace(",", ""))
        ]
        self.current_price_offer: float = self.seller_price_offers[0]

        self.negotiation_completed = False
        self.negotiation_result: str = None  # "accepted" | "rejected" | "max_turns_reached"

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def format_buyer_prompt(self) -> list:
        """Build the full message list for the buyer agent."""
        product = self.product_data
        products_info = (
            f"- {product['Product Name']}:\n"
            f"  Retail Price: {product['Retail Price']}\n"
            f"  Features: {product['Features']}\n"
        )

        messages = [
            {
                "role": "system",
                "content": f"""
Background:
You are a professional negotiation assistant tasked with purchasing a product. Your goal is to negotiate the best possible price for the product, aiming to complete the transaction at the lowest possible price.

Product Information:
{products_info}

Your Budget:
- You have a maximum budget of ${self.budget:.2f} for this purchase.
- Do not exceed this budget under any circumstances.

Goal:
- Negotiate to obtain the product at the lowest possible price
- Use effective negotiation strategies to achieve the best deal

Constraints:
- [IMPORTANT] You must not exceed your budget, otherwise you should reject the offer and say you cannot afford it.

Guidelines:
1. Keep your responses natural and conversational
2. Respond with a single message only
3. Keep your response concise and to the point
4. Don't reveal your internal thoughts or strategy
5. Do not show any bracket about unknown message, like [Your Name]. This is a real conversation between a buyer and a seller.
6. Make your response as short as possible, but do not lose any important information.

Remember: This is a professional negotiation. Your primary goal is to secure the product at the lowest possible price within your budget.""",
            }
        ]

        # Inject conversation history (skip the first buyer intro, which is already sent)
        for turn in self.conversation_history[1:]:
            if turn["speaker"] == "Seller":
                messages.append({"role": "user",      "content": turn["message"]})
            else:
                messages.append({"role": "assistant", "content": turn["message"]})

        return messages

    def format_seller_prompt(self) -> list:
        """Build the full message list for the seller agent."""
        product = self.product_data
        products_info = (
            f"- {product['Product Name']}:\n"
            f"  Retail Price: {product['Retail Price']}\n"
            f"  Wholesale Price: {product['Wholesale Price']}\n"
            f"  Features: {product['Features']}\n"
        )

        messages = [
            {
                "role": "system",
                "content": f"""
Background:
You are a professional sales assistant tasked with selling a product. Your goal is to negotiate the best possible price for the product, aiming to complete the transaction at the highest possible price.

Product Information:
{products_info}

Goal:
- Negotiate to sell the product at the highest possible price
- Use effective negotiation strategies to maximize your profit

Constraints:
- [IMPORTANT] You must not sell below the Wholesale Price

Guidelines:
1. Keep your responses natural and conversational
2. Respond with a single message only
3. Keep your response concise and to the point
4. Don't reveal your internal thoughts or strategy
5. Do not show any bracket about unknown message, like [Your Name]. This is a real conversation between a buyer and a seller.
6. Make your response as short as possible, but do not lose any important information.

Remember: This is a professional negotiation. Your primary goal is to secure the highest possible price, but you must not go below the Wholesale Price.""",
            }
        ]

        for turn in self.conversation_history:
            if turn["speaker"] == "Buyer":
                messages.append({"role": "user",      "content": turn["message"]})
            else:
                messages.append({"role": "assistant", "content": turn["message"]})

        return messages

    # ------------------------------------------------------------------
    # Price extraction
    # ------------------------------------------------------------------

    def extract_price_from_seller_message(self, seller_message: str) -> float | None:
        """
        Ask the summary model to extract the seller's current price offer.
        Returns a float, or None if no price was offered this turn.
        """
        prompt = f"""Extract the price offered by the seller in the following message.
Return only the numerical price (with currency symbol) if there is a clear price offer.
If there is no clear price offer, return 'None'.

IMPORTANT: Only focus on the price of the product itself. Ignore any prices for add-ons like insurance, warranty, gifts, or accessories. Only extract the current offer price for the main product.

Here are some examples:

Example 1:
Seller's message: I can offer you this car for $25000, which is a fair price considering its features.
Price: $25000

Example 2:
Seller's message: Thank you for your interest in our product. Let me know if you have any specific questions about its features.
Price: None

Example 3:
Seller's message: I understand your budget constraints, but the best I can do is $22900 and with giving you a $3000 warranty.
Price: $22900

Example 4:
Seller's message: I can sell it to you for $15500. We also offer an extended warranty for $1200 if you're interested.
Price: $15500

Now for the current message, please STRICTLY ONLY return the price with $ symbol, no other text:

Seller's message:
{seller_message}

Price:"""

        raw = (self.summary_model.get_response(prompt) or "").strip()

        if "None" in raw:
            return None

        price_match = re.search(r'\$([0-9,]+(\.[0-9]+)?)', raw)
        if price_match:
            try:
                return float(price_match.group(1).replace(",", ""))
            except ValueError:
                pass

        logger.warning(f"Could not extract price from: '{raw}'")
        return None

    # ------------------------------------------------------------------
    # Negotiation state evaluation
    # ------------------------------------------------------------------

    def evaluate_negotiation_state(self) -> bool:
        """
        Ask the summary model whether the latest buyer message signals
        ACCEPTANCE, REJECTION, or CONTINUE.

        Returns True if the negotiation is now finished.
        """
        latest_buyer = next(
            (t["message"] for t in reversed(self.conversation_history) if t["speaker"] == "Buyer"),
            None,
        )
        latest_seller = next(
            (t["message"] for t in reversed(self.conversation_history) if t["speaker"] == "Seller"),
            "No response yet",
        )

        if not latest_buyer:
            return False

        evaluation_prompt = f"""
You are evaluating if the buyer's latest response indicates agreement to a deal.

Buyer's latest message: "{latest_buyer}"
Seller's latest message: "{latest_seller}"

Determine if the buyer's response indicates:
A. ACCEPTANCE - The buyer has clearly agreed to the deal
B. REJECTION  - The buyer has clearly rejected the deal or indicated they cannot proceed
C. CONTINUE   - The buyer wants to continue negotiating

Consider the following in your analysis:
- Has the buyer explicitly agreed to purchase at the offered price?
- Has the buyer explicitly rejected the offer or indicated they're walking away?
- Has the buyer indicated they cannot afford the price?
- Is the buyer still asking questions or making counter-offers?

Please strictly output a single line containing just one of: ACCEPTANCE, REJECTION, or CONTINUE.
"""

        evaluation = (self.summary_model.get_response(evaluation_prompt) or "").strip()

        if "ACCEPTANCE" in evaluation:
            self.negotiation_completed = True
            self.negotiation_result = "accepted"
            return True
        elif "REJECTION" in evaluation:
            self.negotiation_completed = True
            self.negotiation_result = "rejected"
            return True
        else:
            return False

    # ------------------------------------------------------------------
    # Main negotiation loop
    # ------------------------------------------------------------------

    def run_negotiation(self) -> list:
        """
        Execute the full negotiation.

        Turn structure:
          1. Buyer sends opening message (generated once before the loop)
          2. Loop: Seller responds → price extracted → Buyer responds → state evaluated
          3. Stop on ACCEPTANCE, REJECTION, or max_turns reached
        """
        print("\nStarting negotiation...")
        print("-" * 50)

        # --- Buyer opening ---
        intro_prompt = f"""You are a professional negotiation assistant aiming to purchase a product at the best possible price.
Your task is to start the conversation naturally without revealing your role as a negotiation assistant.

Please write a short and friendly message to the seller that:
1. Expresses interest in the product and asks about the possibility of negotiating the price
2. Sounds natural, polite, and engaging.

Avoid over-explaining — just say "Hello" to start and smoothly lead into your interest.

Product: {self.product_data['Product Name']}
Retail Price: {self.product_data['Retail Price']}
Features: {self.product_data['Features']}
Your maximum budget for this purchase is ${self.budget:.2f}.

Keep the message concise and focused on opening the negotiation."""

        buyer_intro = self.buyer_model.get_response(intro_prompt)
        self.conversation_history.append({"speaker": "Buyer", "message": buyer_intro})
        print(f"\n[Opening] Buyer: {buyer_intro}")

        # --- Main loop ---
        turn_count = 1
        while turn_count <= self.max_turns:

            # Seller turn
            seller_messages  = self.format_seller_prompt()
            seller_response  = self.seller_model.get_chat_response(seller_messages)
            self.conversation_history.append({"speaker": "Seller", "message": seller_response})
            print(f"\n[Turn {turn_count}] Seller: {seller_response}")

            # Price tracking
            price_offer = self.extract_price_from_seller_message(seller_response)
            if price_offer is not None:
                self.current_price_offer = price_offer
            # Always keep the list long enough
            while len(self.seller_price_offers) <= turn_count:
                self.seller_price_offers.append(self.current_price_offer)
            self.seller_price_offers[turn_count] = self.current_price_offer
            print(f"[Turn {turn_count}] Price on record: ${self.current_price_offer:.2f}")

            # Buyer turn
            buyer_messages  = self.format_buyer_prompt()
            buyer_response  = self.buyer_model.get_chat_response(buyer_messages)
            self.conversation_history.append({"speaker": "Buyer", "message": buyer_response})
            print(f"\n[Turn {turn_count}] Buyer: {buyer_response}")

            # Termination check
            if self.evaluate_negotiation_state():
                print(f"\nNegotiation concluded: {self.negotiation_result}")
                break

            self.completed_turns = turn_count  # record before incrementing
            turn_count += 1

        if not self.negotiation_completed:
            self.completed_turns = self.max_turns  # cap at limit
            self.negotiation_completed = True
            self.negotiation_result = "max_turns_reached"
            print("\nReached maximum turns without a natural conclusion.")

        print("\n" + "-" * 50)
        print(f"Result       : {self.negotiation_result}")
        print(f"Turns taken  : {self.completed_turns}")
        print(f"Final price  : ${self.current_price_offer:.2f}")

        return self.conversation_history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_conversation(self, output_dir: str) -> None:
        """Save the full negotiation record as a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, f"product_{self.product_id}_exp_{self.experiment_num}.json"
        )

        output_data = {
            "product_id":            self.product_id,
            "experiment_num":        self.experiment_num,
            "product_data":          self.product_data,
            "conversation_history":  self.conversation_history,
            "seller_price_offers":   self.seller_price_offers,
            "budget":                self.budget,
            "budget_scenario":       self.budget_scenario,
            "completed_turns":       self.completed_turns,
            "negotiation_completed": self.negotiation_completed,
            "negotiation_result":    self.negotiation_result,
            "models": {
                "buyer":   self.buyer_model_name,
                "seller":  self.seller_model_name,
                "summary": self.summary_model_name,
            },
            "parameters": {
                "max_turns": self.max_turns,
            },
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved → {output_file}")
