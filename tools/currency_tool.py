# tools/currency_tool.py
# Currency Exchange API Integration
# Converts university fees to student's home currency
# Makes costs more relatable for international students

import os
import requests
from dotenv import load_dotenv

load_dotenv()

class CurrencyTool:
    """
    Currency conversion tool using FreeCurrencyAPI.
    
    Converts tuition fees and living costs from USD
    to any currency the student wants.
    
    For Pakistani students: USD → PKR
    For Indian students: USD → INR
    For Nigerian students: USD → NGN
    etc.
    
    Makes costs REAL and relatable for students.
    """
    
    def __init__(self):
        # Using Open Exchange Rates - completely free, no key needed
        self.base_url = "https://open.er-api.com/v6/latest"
        self.enabled  = True
        print("✅ Currency Tool initialized")
        
        # Cache exchange rates
        # So we don't call API repeatedly
        self._rates_cache = {}
    
    # ─────────────────────────────────────────
    # GET EXCHANGE RATES
    # ─────────────────────────────────────────
    
    def get_exchange_rates(self, base_currency: str = "USD") -> dict:
        """Gets latest exchange rates - no API key needed."""
        
        # Return cached rates if available
        if base_currency in self._rates_cache:
            return self._rates_cache[base_currency]
        
        try:
            url      = f"{self.base_url}/{base_currency}"
            response = requests.get(url, timeout=10)
            data     = response.json()
            
            if data.get("result") == "success":
                result = {
                    "success": True,
                    "base":    base_currency,
                    "rates":   data["rates"],
                }
                # Cache it
                self._rates_cache[base_currency] = result
                return result
            else:
                return {"success": False, "rates": {}}
                
        except Exception as e:
            print(f"⚠️ Currency API error: {e}")
            return {"success": False, "rates": {}}
    
    # ─────────────────────────────────────────
    # CONVERT AMOUNT
    # ─────────────────────────────────────────
    
    def convert(self, amount: float, from_currency: str = "USD",
                 to_currency: str = "PKR") -> dict:
        """
        Converts an amount from one currency to another.
        
        Args:
            amount:        amount to convert
            from_currency: source currency (default USD)
            to_currency:   target currency (default PKR)
        
        Returns:
            dict with converted amount and rate
        """
        
        if not self.enabled:
            return {
                "success":        False,
                "original":       amount,
                "converted":      None,
                "currency":       to_currency,
            }
        
        rates_data = self.get_exchange_rates(from_currency)
        
        if not rates_data["success"]:
            return {
                "success":  False,
                "message":  "Could not get exchange rates",
            }
        
        rates = rates_data["rates"]
        
        if to_currency not in rates:
            return {
                "success":  False,
                "message":  f"Currency {to_currency} not supported",
            }
        
        rate      = rates[to_currency]
        converted = amount * rate
        
        return {
            "success":        True,
            "original":       amount,
            "original_currency": from_currency,
            "converted":      round(converted, 2),
            "currency":       to_currency,
            "rate":           rate,
            "formatted":      self._format_amount(
                converted, to_currency
            ),
        }
    
    # ─────────────────────────────────────────
    # UNIVERSITY COST BREAKDOWN
    # ─────────────────────────────────────────
    
    def get_university_cost_breakdown(
        self,
        tuition_usd: float,
        living_cost_usd: float,
        target_currency: str = "PKR",
        duration_years: int = 2
    ) -> dict:
        """
        Creates a complete cost breakdown for a university.
        Shows all costs in both USD and target currency.
        
        Args:
            tuition_usd:     annual tuition in USD
            living_cost_usd: estimated annual living cost in USD
            target_currency: currency to convert to
            duration_years:  program duration
        
        Returns:
            dict with complete cost breakdown
        """
        
        # Convert individual amounts
        tuition_converted = self.convert(
            tuition_usd, "USD", target_currency
        )
        living_converted  = self.convert(
            living_cost_usd, "USD", target_currency
        )
        
        annual_total_usd  = tuition_usd + living_cost_usd
        total_program_usd = annual_total_usd * duration_years
        
        total_converted   = self.convert(
            total_program_usd, "USD", target_currency
        )
        
        breakdown = {
            "success":          True,
            "currency":         target_currency,
            "annual": {
                "tuition_usd":  tuition_usd,
                "living_usd":   living_cost_usd,
                "total_usd":    annual_total_usd,
            },
            "total_program": {
                "years":        duration_years,
                "total_usd":    total_program_usd,
            }
        }
        
        # Add converted amounts if available
        if tuition_converted["success"]:
            breakdown["annual"]["tuition_converted"] = \
                tuition_converted["converted"]
            breakdown["annual"]["tuition_formatted"] = \
                tuition_converted["formatted"]
        
        if living_converted["success"]:
            breakdown["annual"]["living_converted"] = \
                living_converted["converted"]
            breakdown["annual"]["living_formatted"] = \
                living_converted["formatted"]
        
        if total_converted["success"]:
            breakdown["total_program"]["total_converted"] = \
                total_converted["converted"]
            breakdown["total_program"]["total_formatted"] = \
                total_converted["formatted"]
            breakdown["rate"] = total_converted["rate"]
        
        return breakdown
    
    def format_cost_summary(self, university_name: str,
                             tuition_usd: float,
                             target_currency: str = "PKR",
                             duration_years: int = 2) -> str:
        """
        Creates a formatted cost summary string for display.
        
        Args:
            university_name: name of university
            tuition_usd:     annual tuition in USD
            target_currency: currency to show
            duration_years:  program length
        
        Returns:
            formatted cost string
        """
        
        # Estimate living costs by country context
        living_estimate = 10000  # default $10k/year
        
        breakdown = self.get_university_cost_breakdown(
            tuition_usd,
            living_estimate,
            target_currency,
            duration_years
        )
        
        if not breakdown["success"]:
            return f"  💰 Tuition: ${tuition_usd:,}/year"
        
        lines = [
            f"  💰 Annual Tuition:  "
            f"${tuition_usd:,} USD",
        ]
        
        if breakdown["annual"].get("tuition_formatted"):
            lines.append(
                f"                     "
                f"≈ {breakdown['annual']['tuition_formatted']}"
            )
        
        if breakdown["total_program"].get("total_formatted"):
            lines.append(
                f"  💵 Total Program "
                f"({duration_years} years): "
                f"${breakdown['total_program']['total_usd']:,} USD"
            )
            lines.append(
                f"                     "
                f"≈ {breakdown['total_program']['total_formatted']}"
            )
        
        return "\n".join(lines)
    
    def _format_amount(self, amount: float, 
                        currency: str) -> str:
        """Formats currency amount with symbol."""
        
        symbols = {
            "PKR": "₨",
            "INR": "₹",
            "GBP": "£",
            "EUR": "€",
            "CAD": "CA$",
            "AUD": "AU$",
            "NGN": "₦",
            "USD": "$",
        }
        
        symbol = symbols.get(currency, currency)
        
        # Format with commas
        if amount >= 1_000_000:
            formatted = f"{amount/1_000_000:.2f}M"
        elif amount >= 1_000:
            formatted = f"{amount:,.0f}"
        else:
            formatted = f"{amount:.2f}"
        
        return f"{symbol}{formatted} {currency}"